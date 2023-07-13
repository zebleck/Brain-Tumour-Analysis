extends CustomControl

class_name ListItem

@onready var _predictionName : Label = $Panel/VBoxContainer/predictionName
@onready var _predictionValue : ProgressBar = $Panel/VBoxContainer/predictionValue

# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func initalise(prediction : Array):
	_predictionName.text = prediction[0]
	_predictionValue.value = prediction[1]
	pass
