extends CustomControl

@onready var _confirmButton = $Panel/CancerContainer/ConfirmButton
@onready var _startAgainButton = $StartAgain
@onready var _panel = $Panel

signal confirm_button_pressed
signal start_again_button_pressed

# Called when the node enters the scene tree for the first time.
func _ready():
	_confirmButton.connect("pressed", _on_confirm_button_pressed)
	_startAgainButton.connect("pressed", _on_startAgainButton_pressed)
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func _on_confirm_button_pressed():
	confirm_button_pressed.emit()
	_panel.hide()
	pass
	
func _on_startAgainButton_pressed():
	start_again_button_pressed.emit()
	pass

func reload():
	_panel.show()
