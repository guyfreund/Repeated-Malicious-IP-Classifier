import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def prepare_data_for_model(df):
    for col in ["client.size", "client.fingerprint", "ip", "data.src_ip"]:
        if col in df.columns:
            del df[col]
