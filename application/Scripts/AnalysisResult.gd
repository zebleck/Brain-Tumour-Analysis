extends Node

class_name AnalysisResult

var original : ImageTexture
var visualization : ImageTexture
var batch : ImageTexture
var lime : ImageTexture

var metaInfo : String
var prediction : Array
var allPredictions : Array
var score : String
var score_road : String

func _init(original : ImageTexture, visualization : ImageTexture, batch : ImageTexture, lime : ImageTexture, metaInfo : String, prediction : Array, allPredictions : Array, score : String, score_road : String):
	self.original = original
	self.visualization = visualization
	self.batch = batch
	self.lime = lime
	self.metaInfo = metaInfo
	self.prediction = prediction
	self.allPredictions = allPredictions
	self.score = score
	self.score_road = score_road
	pass
# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
