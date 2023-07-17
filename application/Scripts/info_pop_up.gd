extends CustomControl

class_name InfoPopUp

@onready var _originalInfoText : RichTextLabel = $Panel/BasicContainer/Control/OriginalInfoText
@onready var _visualizationInfoText : RichTextLabel = $Panel/BasicContainer/Control/VisualizationInfoText
@onready var _batchInfoText : RichTextLabel = $Panel/BasicContainer/Control/BatchInfoText
@onready var _limeInfoText : RichTextLabel = $Panel/BasicContainer/Control/LimeInfoText

@onready var _closeButton : Button = $Panel/BasicContainer/closeButton

# Called when the node enters the scene tree for the first time.
func _ready():
	_closeButton.connect("pressed", _close_Button_pressed)
	pass # Replace with function body.

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func showInfoBox():
	self.show()
	pass

func activateInfoText(type : String):
	
	_originalInfoText.hide()
	_visualizationInfoText.hide()
	_batchInfoText.hide()
	_limeInfoText.hide()
	
	match type:
		"original":
			_originalInfoText.show()
		"visualization":
			_visualizationInfoText.show()
		"batch":
			_batchInfoText.show()
		"lime":
			_limeInfoText.show()
	pass

func _close_Button_pressed():
	self.hide()
	pass
