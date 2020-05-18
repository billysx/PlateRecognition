from xml.etree.ElementTree import parse
import dicttoxml
from xml.dom.minidom import parseString
import os

# def writeXML(i):
# 	domTree = parse(f"../data/res/{i}.xml")
# 	# 文档根元素
# 	rootNode = domTree.documentElement

# 	# 新建一个customer节点
# 	customer_node = domTree.createElement("customer")
# 	customer_node.setAttribute("ID", "C003")

# 	# 创建name节点,并设置textValue
# 	name_node = domTree.createElement("name")
# 	name_text_value = domTree.createTextNode("kavin")
# 	name_node.appendChild(name_text_value)  # 把文本节点挂到name_node节点
# 	customer_node.appendChild(name_node)

# 	# 创建phone节点,并设置textValue
# 	phone_node = domTree.createElement("phone")
# 	phone_text_value = domTree.createTextNode("32467")
# 	phone_node.appendChild(phone_text_value)  # 把文本节点挂到name_node节点
# 	customer_node.appendChild(phone_node)

# 	# 创建comments节点,这里是CDATA
# 	comments_node = domTree.createElement("comments")
# 	cdata_text_value = domTree.createCDATASection("A small but healthy company.")
# 	comments_node.appendChild(cdata_text_value)
# 	customer_node.appendChild(comments_node)

# 	rootNode.appendChild(customer_node)

# 	with open('added_customer.xml', 'w') as f:
# 		# 缩进 - 换行 - 编码
# 		domTree.writexml(f, addindent='  ', encoding='utf-8')


for i in range(1, 101):
	if not os.path.exists("../data/Plate_dataset/AC/test/xml_pred"):
		os.makedirs("../data/Plate_dataset/AC/test/xml_pred")
	with open(f"../seg/{i}/pred.txt","r") as f:
		data = []
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
	print(prettyxml)

	f = open(f'../data/Plate_dataset/AC/test/xml_pred/{i}.xml','w',encoding='utf-8')
	f.write(prettyxml)


