from gigapath.pipeline import tile_one_slide
import os


local_dir = 'WSI/GPI'
slide_path = os.path.join(local_dir, "J202215433.ndpi")

save_dir = os.path.join(local_dir, 'preprocessing/')

print("NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides. Please make sure to use the appropriate level for the 0.5 MPP")
tile_one_slide(slide_path, save_dir=save_dir, level=0)

print("NOTE: tiling dependency libraries can be tricky to set up. Please double check the generated tile images.")