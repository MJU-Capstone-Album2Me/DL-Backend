import os.path
import logging
import torch

from utils import utils_logger
from utils import utils_image as util
from models.network_rrdbnet import RRDBNet as net
from myutils import converter, resizer, cropper


def deeplearn():

    utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
    logger = logging.getLogger('blind_sr_log')

#    print(torch.__version__)               # pytorch version
#    print(torch.version.cuda)              # cuda version
#    print(torch.backends.cudnn.version())  # cudnn version
    
    ### 이미지 포맷 변환 후 height를 고정해 blind sr
    directory = "../datasets/sample_images"
    ##print(params.directory)
    converter(directory)
    resizer(os.path.join(directory+"_convert"), 1024)

    # -dir, -m
    testsets = '../datasets'       # fixed, set path of testsets
    testset_Ls = [os.path.join(directory+"_convert_resize")]  # ['RealSRSet','DPED']

    #model_names = ['RRDB','ESRGAN','FSSR_DPED','FSSR_JPEG','RealSR_DPED','RealSR_JPEG']
    model_names = ['BSRGANx2']    # 'BSRGANx2' for scale factor 2 or 'BSRGAN'
    #model_names = [params.model]
    print(model_names)


    save_results = True
    sf = 4

    device = torch.device("cpu")
    if(torch.cuda.is_available()):
        device = torch.device("cuda:0")

    print("device:",device)
    for model_name in model_names:
        # set scale factor
        if model_name in ['BSRGANx2']:
            sf = 2
        model_path = os.path.join('model_zoo', model_name+'.pth')   ### set model path
        # path = "model_zoo/BSRGAN.pth" ### x4 baseline
        # path = "model_zoo/BSRGANx2.pth" ### x2 baseline
        # path = "superresolution/bsrgan_x4_gan/models/5000_E.pth"
        # model_path = os.path.join(path)   ### set model path
        logger.info('{:>16s} : {:s}'.format('Model Name', model_name))

        # torch.cuda.set_device(0)      # set GPU ID
        #logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
        torch.cuda.empty_cache()

        # --------------------------------
        # define network and load model
        # --------------------------------
        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network

        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

	# check model parameter
        ### print(summary(model, (3, 1024, 1024))) # torchsummary
        torch.cuda.empty_cache()

        for testset_L in testset_Ls:

            L_path = os.path.join(testsets, testset_L)
            #E_path = os.path.join(testsets, testset_L+'_'+model_name)
            E_path = os.path.join(testsets, testset_L+'_results_x'+str(sf)+"_"+model_path.split("/")[-1])
            util.mkdir(E_path)

            logger.info('{:>16s} : {:s}'.format('Input Path', L_path))
            logger.info('{:>16s} : {:s}'.format('Output Path', E_path))
            idx = 0

            for img in util.get_image_paths(L_path):

                # --------------------------------
                # (1) img_L
                # --------------------------------
                idx += 1
                img_name, ext = os.path.splitext(os.path.basename(img))
                logger.info('{:->4d} --> {:<s} --> x{:<d}--> {:<s}'.format(idx, model_name, sf, img_name+ext))

                img_L = util.imread_uint(img, n_channels=3)
                img_L = util.uint2tensor4(img_L)
                img_L = img_L.to(device)

                # --------------------------------
                # (2) inference
                # --------------------------------
                img_E = model(img_L)

                # --------------------------------
                # (3) img_E
                # --------------------------------
                img_E = util.tensor2uint(img_E)
                if save_results:
                    util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+'.png'))

if __name__ == '__main__':

    #parser = argparse.ArgumentParser(description='BSRGAN')

    # setup for inference
    #parser.add_argument("--directory","-dir", type=str, default="../datasets/sample_images",
    #                    help="Which directory to inference")
    #parser.add_argument("--model","-m", type=str, default="BSRGANx2",
    #                    help="which model to inference. BSRGANx2 or BSRGAN")
    #parser.add_argument('--disable_cuda', action='store_true',
    #                    help='Disable CUDA')
    #params = parser.parse_args()

    #if not params.disable_cuda and torch.cuda.is_available():
    #    params.device = torch.device("cuda:0")
    #else:
    #    params.device = torch.device("cpu")

    #params.device = torch.device("cpu")
    deeplearn()
