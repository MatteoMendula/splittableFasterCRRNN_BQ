import os
import sys
from PIL import Image
  
# define a function for
# compressing an image
def compressMe(file):
    
      
    # open the image
    picture = Image.open(file)
      
    # Save the picture with desired quality
    # To change the quality of image,
    # set the quality variable at
    # your desired level, The more 
    # the value of quality variable 
    # and lesser the compression
    print("Compressing "+file+"...")
    picture.save("check_"+file, 
                 "JPEG", 
                #  optimize = True, 
                 quality = 70)
    return
  
# Define a main function
def main():
    compressMe("car.jpg")

if __name__ == "__main__":
    main()