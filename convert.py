from source.load import load_from_google_cloud
from source.models import *

if __name__ == '__main__':
    run_name = 'to-ml-register-template_20200217063212_HighResNet_RPRC_HistLoss_64_0.0001_HistLoss_RPRC_temp'
    model = HighResNet()
    load_from_google_cloud(run_name, 17, model)