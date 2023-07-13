extends CustomControl

class_name ImageNameList

@onready var _itemList : ItemList = $ItemList

var _currentSelectedItem : String = "Default"

# Called when the node enters the scene tree for the first time.
func _ready():
	_itemList.connect("item_selected", _on_item_selected)
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func clear():
	_itemList.clear()
	pass

func addItems(imageNames : Array):
	for name in imageNames:
		_itemList.add_item(name)
	pass
	_itemList.sort_items_by_text()
pass

func addItemsPredictions(predictions : Array):
	for prediction in predictions:
		_itemList.add_item(prediction[0] + " | " + str(prediction[1]))
	pass
	_itemList.sort_items_by_text()
pass

func getSelectedItemText() -> String:
	var temp = _currentSelectedItem
	_currentSelectedItem = "DEFAULT"
	return temp

func _on_item_selected(index : int):
	_currentSelectedItem = _itemList.get_item_text(index)
	pass
