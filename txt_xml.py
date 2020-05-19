from xml.etree.ElementTree import parse
import dicttoxml
from xml.dom.minidom import parseString
import os

for i in range(1, 101):
	if not os.path.exists("../data/Plate_dataset/AC/test/xml_pred"):
		os.makedirs("../data/Plate_dataset/AC/test/xml_pred")
	data = []
	with open(f"../seg/{i}/pred.txt","r") as f:
		for line in f:
			data.append(line.strip())
	with open(f"../seg/{i}/bbox.txt","r") as f:
		for line in f:
			data.append(line.strip())
	bbox = data[1].split(",")
	resdict = {}

	resdict["object"] = {}
	resdict["object"]["bndbox"] = {}
	resdict["object"]["bndbox"]["xmin"] = bbox[0]
	resdict["object"]["bndbox"]["ymin"] = bbox[1]
	resdict["object"]["bndbox"]["xmax"] = bbox[2]
	resdict["object"]["bndbox"]["ymax"] = bbox[3]
	resdict["object"]["platetext"] = data[0]

	bxml = dicttoxml.dicttoxml(resdict, custom_root="annotation")
	xml = bxml.decode('utf-8')
	dom = parseString(xml)
	prettyxml = dom.toprettyxml(indent='    ')

	f = open(f'../data/Plate_dataset/AC/test/xml_pred/{i}.xml','w',encoding='utf-8')
	f.write(prettyxml)


