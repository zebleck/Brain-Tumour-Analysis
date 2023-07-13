extends Control

@onready var _welcomeContainer = $Panel/WelcomeContainer
@onready var _selectionContainer = $Panel/SelectionContainer
@onready var _startContainer = $Panel/StartContainer

@onready var _welcomNameText : Label = $Panel/WelcomeContainer/WelcomeNameText
@onready var _welcomInfoText : Label  = $Panel/WelcomeContainer/WelcomeInfoText
@onready var _selectionInfoText : Label  = $Panel/SelectionContainer/SelectionInfoText

@onready var _startButton : Button = $Panel/WelcomeContainer/StartButton
@onready var _selectButton : Button  = $Panel/SelectionContainer/SelectButton
@onready var _classifictaionStartButton : Button  = $Panel/StartContainer/ClassificationButton
@onready var _selectOtherButton : Button  = $Panel/StartContainer/SelectOtherButton

@onready var _pathInput : InputStandard = $Panel/WelcomeContainer/PathInput


signal start_button_pressed(filePath : String)
signal select_button_pressed
signal classification_button_pressed
signal select_other_button_pressed

# Called when the node enters the scene tree for the first time.
func _ready():
	_startButton.connect("pressed", _start_button_pressed)
	_selectButton.connect("pressed", _select_button_pressed)
	_classifictaionStartButton.connect("pressed", _classification_button_pressed)
	_selectOtherButton.connect("pressed", _select_other_button_pressed)
	
	_selectButton.hide()
	_classifictaionStartButton.hide()
	_selectOtherButton.hide()
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func _start_button_pressed():
	
	start_button_pressed.emit(_pathInput.getFilePath())
	pass

func _select_button_pressed():
	select_button_pressed.emit()
	pass

func _classification_button_pressed():
	classification_button_pressed.emit()
	pass

func _select_other_button_pressed():
	select_other_button_pressed.emit()
	pass

func showWelcome(visible : bool):
	if(visible):
		_welcomeContainer.show()
		return
	_welcomeContainer.hide()
	pass

func showSelection(visible : bool):
	if(visible):
		var tween = _selectionContainer.create_tween()
		tween.tween_property(_selectionContainer, "modulate", Color(1,1,1,1), 1.5).from(Color(1,1,1,0)).set_trans(Tween.TRANS_CUBIC).set_ease(Tween.EASE_OUT)
		tween.tween_callback(func(): _selectButton.show())
		_selectionContainer.show()
		return
	_selectionContainer.hide()
	_selectButton.hide()
	pass

func showStart(visible : bool):
	if(visible):
		_startContainer.show()
		var tween = _startContainer.create_tween()
		tween.tween_property(_startContainer, "modulate", Color(1,1,1,1), 1.5).from(Color(1,1,1,0)).set_trans(Tween.TRANS_CUBIC).set_ease(Tween.EASE_OUT)
		tween.tween_callback(
		func(): 
			_classifictaionStartButton.show()
			_selectOtherButton.show()
			pass)
		return
	_startContainer.hide()
	_classifictaionStartButton.hide()
	_selectOtherButton.hide()
	pass
