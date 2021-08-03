import os
import cv2
import xml.etree.ElementTree as ET
import argparse

def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars

def copy_data(xml_path, image_path):
    xmllist = [item for item in os.listdir(xml_path) if item.endswith(".xml")]
    for xml in xmllist:
        image_name = "{}_{}".format(xml[:-4], args.device)
        
        # xml 
        xml_file = os.path.join(xml_path, xml)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        root[1].text = image_name
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category != "piglet":
                root.remove(obj)
        tree.write("{}/xmls/{}.xml".format(args.dataset, image_name))
        
        # image
        img = cv2.imread(os.path.join(image_path, xml[:-4]+".jpg"))
        copy_img = img.copy()
        cv2.imwrite("{}/images/{}.jpg".format(args.dataset, image_name), copy_img)
      
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy data from Labeled image.")
    parser.add_argument("--xml_dir", default="../../../../Labeled image/piglet behavior/detection/rotated_roLabelimg", help="Directory path to xml files.", type=str)
    parser.add_argument("--image_dir", default="../../../../Crop images", help="Directory path to image files.", type=str)
    parser.add_argument("--device", default="rpi2", help="Which device to use.", type=str)
    parser.add_argument("--time", default="Day", help="", type=str)
    parser.add_argument("--dataset", default="train", help="", type=str)
    args = parser.parse_args()
    
    xml_path = os.path.join(args.xml_dir, args.device, args.time)
    image_path = os.path.join(args.image_dir, args.device, args.time)
#    xml_path = '../../../../Datasets/Detection/Augmentation/Rotation/rpi3/Annotation'
#    image_path = '../../../../Datasets/Detection/Augmentation/Rotation/rpi3/Image'
    copy_data(xml_path, image_path)