extends CustomControl

class_name CustomWarningMessage

@onready var _headline : Label = $Panel/BasicContainer/Label
@onready var _error : Label = $Panel/BasicContainer/Error
@onready var _info : Label = $Panel/BasicContainer/Info
@onready var _info2 : Label = $Panel/BasicContainer/Info2

@onready var _closeButton : Button = $Panel/BasicContainer/closeButton
@onready var _quitButton : Button = $Panel/BasicContainer/quitButton

# Called when the node enters the scene tree for the first time.
func _ready():
	_closeButton.connect("pressed", _close_Button_pressed)
	_quitButton.connect("pressed", _quit_Button_pressed)
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func showWarning(headline : String, error : String, info : String, info2 : String, isQuitMessage : bool):
	
	if(isQuitMessage):
		_quitButton.show()
	else:
		_quitButton.hide()
	
	_headline.text = headline
	_error.text = error
	_info.text = info
	_info2.text = info2
	
	self.show()
	
	pass
	
func _close_Button_pressed():
	self.hide()
	pass

func _quit_Button_pressed():
	get_tree().quit()
	pass
