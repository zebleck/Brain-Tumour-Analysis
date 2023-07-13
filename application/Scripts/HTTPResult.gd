extends HTTPRequestCustom

class_name HTTPResult

var _header = ["Content-Type: application/json; charset=UTF-8"]
var _currentFileName = "DEFAULT"

signal result_loadet(result : AnalysisResult)
signal request_failed(error : String, info : String )


func getResultWithImageName(fileName : String):
	_currentFileName = fileName
	self.add_to_queue(Global.urlWithImageNameVariable(fileName), _header, HTTPClient.METHOD_GET, "", _handleResult)
	pass

func getArchiveWithFolderName(folderName : String):
	_currentFileName = folderName
	self.add_to_queue(Global.urlWithFolderNameArchiveVariable(folderName), _header, HTTPClient.METHOD_GET, "", _handleResult)
	pass

func getResultWithImage(filepath : String):
	var image = Image.new()
	image.load("user://Histo-Pictures/" + filepath)
	var raw_data = image.get_data()
	var encodedImage = Marshalls.raw_to_base64(raw_data)
	
	var json_string = JSON.stringify({
		'image' : raw_data
	})
	
	self.add_to_queue(Global.urlWithImage, _header, HTTPClient.METHOD_POST, json_string, _handleResult)
	pass

func getResult():
	self.add_to_queue(Global.url, _header, HTTPClient.METHOD_GET, "", _handleResult)
	pass

func _handleResult(result, response_code, header, body):
	if(response_code == 200):
		var json = JSON.parse_string(body.get_string_from_utf8())

		var originalImage = Image.new()
		var visualizationImage = Image.new()
		var batchImage = Image.new()
		var limeImage = Image.new()
		
		var error = originalImage.load_png_from_buffer(Marshalls.base64_to_raw(json.original))#json.visualization))

		if error != OK:
			print("Couldn't load the original image.")
			push_error("Couldn't load the original image.")
			pass
		
		error = visualizationImage.load_png_from_buffer(Marshalls.base64_to_raw(json.visualization))
		
		if error != OK:
			print("Couldn't load the visualization image.")
			push_error("Couldn't load the visualization image.")
			pass
		
		error = batchImage.load_png_from_buffer(Marshalls.base64_to_raw(json.batch_visualizations))
		
		if error != OK:
			print("Couldn't load the batch image.")
			push_error("Couldn't load the batch image.")
			pass
		
		error = limeImage.load_png_from_buffer(Marshalls.base64_to_raw(json.lime_visualization))
		
		if error != OK:
			print("Couldn't load the lime image.")
			push_error("Couldn't load the lime image.")
			pass
		#error = image.save_png("res://editor_result.png")

		#if error != OK:
		#	print("Couldn't save result image.")
		#	push_error("Couldn't save result image.")
		
		var original = ImageTexture.create_from_image(originalImage)
		var visualization = ImageTexture.create_from_image(visualizationImage)
		var batch = ImageTexture.create_from_image(batchImage)
		var lime = ImageTexture.create_from_image(limeImage)
		
		var metaInfo = _currentFileName
		var prediction = json.predictions[0]
		var allPredictions = json.predictions
		var score = json.scores
		var score_road = json.scores_road
		
		result_loadet.emit(AnalysisResult.new(original, visualization, batch, lime, metaInfo, prediction, allPredictions, score, score_road))
		
	else:
		var error = "Error - HTTPResult: " + str(response_code)
		var info = JSON.parse_string(body.get_string_from_utf8())
		
		if(info == null):
			info = "no info"
		
		print(error)
		print(info)
		request_failed.emit(error, info)
		pass
	pass
