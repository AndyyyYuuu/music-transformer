import music21
import os
from xml.etree import ElementTree
import random


# Converts xml to midi and saves at new path
def xml_to_midi(xml_pth, midi_pth):
    converter = music21.converter.parse(xml_pth)
    converter.write("midi", midi_pth)


# Removes all xml tags of `tag` in a folder and saves the new data
def remove_tag(source, destination, root_tag, tag):
    tree = ElementTree.parse(source)
    root = tree.getroot()

    #harmonies_to_remove = root.findall('.//{tag}')
    #print(list(root))
    #print(harmonies_to_remove)
    # Find and remove all <harmony> elements
    # find all <d> nodes
    for node in root.iter(root_tag):
        # find <e> subnodes of <d>
        for subnode in node.iter(tag):
            node.remove(subnode)


    # Write the modified XML to a new file
    tree.write(destination)


def remove_all_chords(source, destination):
    for item in os.listdir(source):
        try:
            remove_tag(os.path.join(source, item), os.path.join(destination, item), "measure", "harmony")
        except ElementTree.ParseError:
            pass


# Removes
def process_xml_folder(source, destination):
    for item in os.listdir(source):
        xml_to_midi(os.path.join(source, item), os.path.join(destination, os.path.splitext(item)[0]+".midi"))


# remove_all_chords("omnibook_xml", "omnibook_xml_nochords")
# process_xml_folder("omnibook_xml_nochords", "omnibook_midi")

def random_sample(source, destination, prop):
    indices = list(range(len(os.listdir(source))))
    print(indices)
    random.shuffle(indices)
    indices = indices[:int(len(indices)*prop)]
    dirs = os.listdir(source)
    for i in indices:
        d = dirs[i]
        os.rename(os.path.join(source, d), os.path.join(destination, d))
