# from model.swin.swin import Swin
from model.swin.swin import EBM_Prior
from model.swin.swin import ResNet_endecoder
from model.swin.swin import SwinDecoder, CNNDecoder

def get_model(option):
    model_name = option['model_name']
    spv_swin_model = SwinDecoder(option['trainsize'])
    # cnn_model = CNNDecoder(option['trainsize'])
    semi_swin_model = SwinDecoder(option['trainsize'])
    
    print("Spv_Swin_Model based on {} have {:.4f}Mb paramerters in total".format(model_name, sum(x.numel()/1e6 for x in spv_swin_model.parameters())))
    print("Semi_Swin_Model based on {} have {:.4f}Mb paramerters in total".format(model_name, sum(x.numel()/1e6 for x in semi_swin_model.parameters())))
    
    # print("CNN_Model based on {} have {:.4f}Mb paramerters in total".format(model_name, sum(x.numel()/1e6 for x in cnn_model.parameters())))
    
    
    return spv_swin_model.cuda(), semi_swin_model.cuda()