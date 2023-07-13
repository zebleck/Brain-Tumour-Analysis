extends CustomControl

class_name ViewportResult

@onready var _originalButton : Button = $ImageSelectionContainer/originalButton
@onready var _visualisationButton : Button = $ImageSelectionContainer/visualizationButton
@onready var _batchButton : Button = $ImageSelectionContainer/batchButton
@onready var _limeButton : Button = $ImageSelectionContainer/limeButton

@onready var _alphaSlider : HSlider = $alphaSlider

@onready var _resultImage : TextureRect = $ResultImage
@onready var _originalImage : TextureRect = $OriginalImage

@onready var _metaLabel : Label = $Panel2/InformationContainer/VBoxContainer/meta
@onready var _predictionLabel : Label = $Panel2/InformationContainer/VBoxContainer/Panel/VBoxContainer/prediction
@onready var _scoreLabel : Label = $Panel2/InformationContainer/VBoxContainer/score
@onready var _scoreRoadLabel : Label = $Panel2/InformationContainer/VBoxContainer/score_road

@onready var _predictionValue : ProgressBar = $Panel2/InformationContainer/VBoxContainer/Panel/VBoxContainer/predictionValue
@onready var _scoreValue : ProgressBar = $Panel2/InformationContainer/VBoxContainer/score_value
@onready var _scoreRoadValue : ProgressBar = $Panel2/InformationContainer/VBoxContainer/score_road_value

@onready var _predictionList = $Panel3/PredictionList

@onready var _infoBox : InfoPopUp = $InfoBox
@onready var _infoButton : Button = $InfoButton

#@onready var _indicatorOriginal : Panel = $ImageSelectionFeedbackContainer/IndicatorOriginal
#@onready var _indicatorVisualization : Panel = $ImageSelectionFeedbackContainer/IndicatorVisualization
#@onready var _indicatorBatch : Panel = $ImageSelectionFeedbackContainer/IndicatorBatch
#@onready var _indicatorLime : Panel = $ImageSelectionFeedbackContainer/IndicatorLime

var _original : ImageTexture
var _visualization : ImageTexture
var _batch : ImageTexture
var _lime : ImageTexture

# Called when the node enters the scene tree for the first time.
func _ready():
	
	_originalButton.connect("pressed", _original_button_pressed)
	_visualisationButton.connect("pressed", _visualization_button_pressed)
	_batchButton.connect("pressed", _batch_button_pressed)
	_limeButton.connect("pressed", _lime_button_pressed)
	
	_alphaSlider.connect("value_changed", _on_alpha_value_changed)
	
	_infoButton.connect("pressed", _info_button_pressed)
	
	#_indicatorVisualization.hide()
	#_indicatorBatch.hide()
	#_indicatorLime.hide()
	pass # Replace with function body.

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func _info_button_pressed():
	_infoBox.showInfoBox()
	pass

func _on_alpha_value_changed(value):
	_resultImage.modulate = Color(1,1,1,value)
	pass

func setResult(result : AnalysisResult):
	_original = result.original
	_originalImage.texture = _original
	_visualization = result.visualization
	_batch = result.batch
	_lime = result.lime
	
	_metaLabel.text = result.metaInfo
	print(result.prediction[0])
	_predictionLabel.text = result.prediction[0]
	print(result.prediction)
	_predictionValue.value = float(result.prediction[1])
	_scoreLabel.text = "score: " + result.score
	_scoreValue.value = abs(float(result.score))
	_scoreRoadLabel.text = "score_road: " + result.score_road
	_scoreRoadValue.value = float(result.score_road)
	
	#_predictionList.addItemsPredictions(result.allPredictions)
	_predictionList.generatePredictionList(result.allPredictions)
	pass

func _original_button_pressed():
	_resultImage.hide()
	_infoBox.activateInfoText("original")
	#_indicatorVisualization.hide()
	#_indicatorBatch.hide()
	#_indicatorLime.hide()
	#_indicatorOriginal.show()
	pass

func _visualization_button_pressed():
	_resultImage.show()
	_resultImage.texture = _visualization
	_infoBox.activateInfoText("visualization")
	#_indicatorVisualization.show()
	#_indicatorBatch.hide()
	#_indicatorLime.hide()
	#_indicatorOriginal.hide()
	pass

func _batch_button_pressed():
	_resultImage.show()
	_resultImage.texture = _batch
	_infoBox.activateInfoText("batch")
	#_indicatorVisualization.hide()
	#_indicatorBatch.show()
	#_indicatorLime.hide()
	#_indicatorOriginal.hide()
	pass

func _lime_button_pressed():
	_resultImage.show()
	_resultImage.texture = _lime
	_infoBox.activateInfoText("lime")
	#_indicatorVisualization.hide()
	#_indicatorBatch.hide()
	#_indicatorLime.show()
	#_indicatorOriginal.hide()
	pass
