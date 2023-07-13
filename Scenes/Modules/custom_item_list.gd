extends CustomControl

class_name CustomItemList

@onready var _list : VBoxContainer = $ScrollContainer/List

# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func generatePredictionList(predictions : Array):
	for prediction in predictions:
		var listItem = Global.create_instance()
		_list.add_child(listItem)
		listItem.initalise(prediction)
	pass
