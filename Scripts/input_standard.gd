extends CustomControl

class_name InputStandard

@onready var _LineEdit : LineEdit = $InputContainer/Background/LineEdit

func getFilePath() -> String:
	return _LineEdit.get_text()

# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
