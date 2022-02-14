import random
from glob import glob
from zero_dce import (
    download_dataset, init_wandb,
    Trainer, plot_result
)
from PIL import Image
from matplotlib import cm
import numpy as np
import sys, getopt

def infer_file(input_file, output_file):
    trainer = Trainer()
    trainer.build_model(pretrain_weights='./pretrained_models/model200_dark_faces.pth')
    image, enhanced = trainer.infer_gpu(input_file, image_resize_factor=1)
    # plot_result(image, enhanced)
    print (enhanced)
    # enhanced.reshape(enhanced.shape[0], enhanced.shape[1])
    im = Image.fromarray((enhanced * 255).astype(np.uint8))
    im.save(output_file)

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print ('main.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('main.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   
   infer_file(inputfile, outputfile)


if __name__ == "__main__":
   main(sys.argv[1:])
