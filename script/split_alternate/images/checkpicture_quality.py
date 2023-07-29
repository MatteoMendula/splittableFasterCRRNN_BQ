import os
import sys
from PIL import Image
  
# define a function for
# compressing an image
def compressMe(file, compression):
    
      
    # open the image
    picture = Image.open(file)
      
    # Save the picture with desired quality
    # To change the quality of image,
    # set the quality variable at
    # your desired level, The more 
    # the value of quality variable 
    # and lesser the compression
    print("Compressing "+file+"...")
    picture.save("matte_"+str(compression)+".jpg", 
                 "JPEG", 
                #  optimize = True, 
                 quality = compression)
    return
  
# Define a main function
def main():
    for el in [50, 30]:
        compressMe("matte.jpg", el)

if __name__ == "__main__":
    main()