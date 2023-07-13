extends CustomControl

class_name Menu

@onready var _newButton : Button = $Panel/HBoxContainer/newButton
@onready var _archiveButton : Button = $Panel/HBoxContainer/archiveButton
@onready var _exitButton : Button = $Panel/HBoxContainer/exitButton

signal _on_new_button_pressed
signal _on_archive_Button_pressed
signal _on_quit_button_pressed

# Called when the node enters the scene tree for the first time.
func _ready():
	
	_newButton.connect("pressed", _new_button_pressed)
	_archiveButton.connect("pressed", _archive_button_pressed)
	_exitButton.connect("pressed", closeApp)
	
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func closeApp():
	_on_quit_button_pressed.emit()
	pass

func _new_button_pressed():
	_on_new_button_pressed.emit()
	pass

func _archive_button_pressed():
	_on_archive_Button_pressed.emit()
	pass
