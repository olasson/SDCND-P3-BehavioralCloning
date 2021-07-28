from moviepy.editor import ImageSequenceClip

# Custom imports
from os import listdir
from os.path import join as path_join
from os.path import splitext

IMAGE_EXT = ['jpeg', 'gif', 'png', 'jpg']


def make_video(path_images_record, file_path_video_save, fps):
    """
    Create an .mp4 video from a set of images
    
    Inputs
    ----------
    path_images_record: str
        Path to where the images from the recorded run are saved
    file_path_video_save: str
        Path to where the output video will be saved
    fps: int
        Framed per second in the output video
       
    Outputs
    -------
        Saves an .mp4 video at 'file_path_video_save'
    """

    image_list = sorted([path_join(path_images_record, image_file)
                        for image_file in listdir(path_images_record)])
    
    image_list = [image_file for image_file in image_list if splitext(image_file)[1][1:].lower() in IMAGE_EXT]

    clip = ImageSequenceClip(image_list, fps = fps)

    clip.write_videofile(file_path_video_save)