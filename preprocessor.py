from constants import PICKLES_DIR

import json
import pandas as pd
import os
from datetime import datetime
from itertools import zip_longest


class Preprocessor:
    def __init__(self, path_to_json, save_pickles=False, add_label=True):
        self.log('init start')
        self.save_pickles = save_pickles
        if self.save_pickles and not os.path.isdir(PICKLES_DIR):
            os.mkdir(PICKLES_DIR)
        self.add_label = add_label
        self.log('loading json start')
        if os.path.exists(path_to_json):
            with open(path_to_json, 'r') as f:
                self.sessions_json = json.load(f)
            self.log('loading json end')
        else:
            raise Exception(f'{path_to_json} does not exist, could not initialize Preprocessor.')
        self.log('init end')


    def normalize_in_chunks(self) -> pd.DataFrame:
        def grouper(iterable, n, fillvalue=None):
            "Collect data into fixed-length chunks or blocks"
            # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
            args = [iter(iterable)] * n
            return zip_longest(*args, fillvalue=fillvalue)

        self.log('normalize_in_chunks start')
        dfs = []
        chunk_size = 10000
        i = 1

        for json_chunk in grouper(self.sessions_json, chunk_size):
            df = pd.json_normalize(json_chunk)
            if self.save_pickles:
                filename = f"{PICKLES_DIR}/df_{i}.pkl"
                self.log(f"pickeling file {filename}")
                df.to_pickle(filename)
                i += 1
            dfs.append(df)

        final_df = pd.concat(dfs)
        self.log('normalize_in_chunks end')
        return final_df


    def process(self) -> pd.DataFrame:
        """ Preprocess the data - end to end """

        # flatten the json
        df = self.normalize_in_chunks()

        # removing rows without session id
        df = df[df["data.session"].notna()]

        # removing columns which aren't meaningful
        cols_to_remove_new = ['_id.$oid', 'data.src_port', 'data.timestamp', 'data.weather', 'data.port', 'data.size',
                              'data.shasum', 'data.ttylog', 'data.duplicate', 'data.duration', 'data.sen_os',
                              'data.sen_ver',
                              'data.src_os_flavor', 'data.hassh', 'data.hasshAlgorithms', 'data.kexAlgs',
                              'data.keyAlgs',
                              'data.encCS', 'data.macCS', 'data.compCS', 'data.langCS', 'data.username',
                              'data.password',
                              'data.version', 'data.arch', 'data.input', 'data.sensor_geo.$binary.subType',
                              'data.sensor_segment.$binary.subType']
        for col in cols_to_remove_new:
            if col in df.columns:
                del df[col]

        # timing has no meaning, therefore using hot-encoding
        df.loc[df["data.src_uptime_sec"].notna(), "data.src_uptime_sec"] = 1
        df.loc[df["data.src_uptime_sec"].isna(), "data.src_uptime_sec"] = 0

        # sessions aggregation
        sessions_df = self.aggregate_session_data(df)

        # labeling
        if self.add_label:
            self.log(msg="starting add label")
            self.add_label_column(sessions_df)
            self.log(msg="finished add label")

        del sessions_df["data.session"]  # sessions id isn't meaningful anymore

        # transforming into numeric data
        categorical_cols = ["ip", "data.sensor_geo.$binary.base64", "data.sensor_segment.$binary.base64",
                            "data.ip_country", "data.message", "data.sensor", "data.src_ip", "data.sen_dist",
                            "data.src_language", "data.src_os_name"]
        processed_data = self.factorize_df(sessions_df, categorical_cols)

        if processed_data.iloc[-1].isnull().values.any():
            # remove the last row if it contains a nan
            processed_data = processed_data.iloc[:-1]

        # saving the preprocessed data
        if self.save_pickles:
            processed_data.to_pickle(f"{PICKLES_DIR}/factorized_data.pkl")
            self.log(msg="pickled factorized_data")

        return processed_data

    def aggregate_session_data(self, df) -> pd.DataFrame:
        """ Aggregates the raw data into a sample per session """

        self.log('aggregate_session_data start')
        unique_session_ids = df["data.session"].unique()
        self.log(msg=f'number of unique session ids: {len(unique_session_ids)}')

        # types of sessions events
        event_types = ['log.closed', 'client.kex', 'login.success', 'session.closed', 'client.version',
                       'session.params',
                       'command.input', 'session.file_download', 'session.file_download.failed', 'command.failed',
                       'session.file_upload', 'client.var', 'command.success']
        cols = list(df.columns) + event_types  # transforming the columns
        cols.remove("data.eventid")
        cols.remove("data.session")

        sessions_df = pd.DataFrame(columns=cols)

        # columns which all values are unique for all session events
        identical_cols = ["data.session", "ip", "data.ip_country", "data.sensor", "data.src_ip", "data.sen_dist",
                          "data.src_language", "data.src_os_name", "data.src_uptime_sec",
                          "data.sensor_geo.$binary.base64",
                          "data.sensor_segment.$binary.base64", "data.bf"]

        # DataFrame creation
        for i, session_id in enumerate(unique_session_ids):
            # logging
            if i % 100 == 0:
                self.log(msg=f'{i}. Session ID {session_id}')
                if self.save_pickles:
                    sessions_df.to_pickle(f"{PICKLES_DIR}/processed_data_{i}.pkl")

            session_events = df[df["data.session"] == session_id]
            first_session = session_events.iloc[0]
            session = {col: first_session[col] for col in identical_cols}
            session.update({col: 0 for col in event_types})
            event_id_fields = session_events["data.eventid"].value_counts().to_dict()
            session.update(event_id_fields)
            session["data.message"] = session_events["data.message"].to_string()
            sessions_df = sessions_df.append(session, ignore_index=True)

        sessions_df.loc["data.bf"] = sessions_df["data.bf"].astype(int)

        # saving the processed data
        if self.save_pickles:
            sessions_df.to_pickle(f"{PICKLES_DIR}/processed_data.pkl")
            self.log(msg="pickled processed_data")

        self.log('aggregate_session_data end')
        return sessions_df

    def add_label_column(self, df) -> None:
        """ Adding the label - 0 if ip doesn't return in the data, else 1. """

        self.log('add_label_column start')
        df["label"] = 0
        unique_ips = df["ip"].unique()
        self.log(msg=f"number of unique ips: {len(unique_ips)}")

        # label creation
        for i, ip in enumerate(unique_ips):
            # logging
            if i % 100 == 0:
                self.log(msg=f'{i}. IP {ip}')

            sessions = df[df["ip"] == ip]["data.session"].unique()  # all sessions with the same IP
            if len(sessions) != 1:  # IP is repeated in different sessions
                df.loc[df.ip == ip, "label"] = 1

        self.log('add_label_column end')

    def factorize_df(self, df, categorical_cols) -> pd.DataFrame:
        """ Transforms non-numeric values into dummy values """

        self.log('factorize_df start')
        new_df = pd.DataFrame(columns=df.columns)
        for col in categorical_cols:
            new_df[col], _ = df[col].factorize()
        not_categorical_cols = [col for col in df.columns if col not in categorical_cols]
        for col in not_categorical_cols:
            new_df[col] = df[col]
        self.log('factorize_df end')
        return new_df

    @staticmethod
    def log(msg):
        print(f'{str(datetime.now())} Preprocessor: {msg}')
