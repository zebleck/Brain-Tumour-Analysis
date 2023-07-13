extends Control

@onready var _startMenu = $StartMenu
@onready var _loading = $Loading
@onready var _result = $Result

@onready var _httpResult : HTTPResult = $HTTP/HTTPResult
@onready var _httpImageNames : HTTPImageNames = $HTTP/HTTPImageNames

@onready var _resultImage : TextureRect = $ResultImage

@onready var _imageNamesList : ImageNameList = $ImageNamesList

@onready var _initMenu : InitMenu = $initMenu

# Called when the node enters the scene tree for the first time.
func _ready():
	
	Global.httpResult = self._httpResult
	Global.httpImageNames = self._httpImageNames
	
	_startMenu.connect("start_button_pressed", _startMenu_on_start_button_pressed)
	_startMenu.connect("select_button_pressed", _startMenu_on_select_button_pressed)
	_startMenu.connect("classification_button_pressed", _startMenu_on_classification_button_pressed)
	_startMenu.connect("select_other_button_pressed", _startMenu_on_select_other_button_pressed)
	
	_loading.connect("loading_finished", _loading_finished)
	_loading.connect("loading_next", _loading_next)
	
	_result.connect("confirm_button_pressed", _result_on_confirm_button_pressed)
	_result.connect("start_again_button_pressed", _result_on_start_again_button_pressed)
	
	_httpResult.connect("result_loadet", _on_result_loadet)
	_httpImageNames.connect("image_file_names_loadet", _on_names_loadet)
	_httpImageNames.getAllImageFileNames()
	
	_initMenu.initalise()
	
	pass # Replace with function body.

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func _on_names_loadet(names : Array):
	_imageNamesList.addItems(names)
	pass

func _on_result_loadet(texture : ImageTexture):
	_resultImage.texture = texture
	_resultImage.show()
	pass

func _startMenu_on_start_button_pressed(filePath : String):
	#_startMenu.showWelcome(false)
	
	_initMenu.showMenu("start")
	
	#_httpResult.getResultWithImage(filePath)
	#_httpResult.getResult()
	#_httpImageNames.getAllImageFileNames()
	#_httpResult.getResultWithImageName(_imageNamesList.getSelectedItemText())
	
	#_startMenu.showSelection(true)
	pass

func _startMenu_on_select_button_pressed():
	_startMenu.showSelection(false)
	_startMenu.showStart(true)
	pass

func _startMenu_on_classification_button_pressed():
	_startMenu.hide()
	_startMenu.showStart(false)
	_loading.show()
	_loading.startLoading()
	pass

func _startMenu_on_select_other_button_pressed():
	_startMenu.showSelection(true)
	_startMenu.showStart(false)
	pass

func _loading_finished():
	_loading.next()
	pass

func _loading_next():
	_result.show()
	pass

func _result_on_confirm_button_pressed():
	pass

func _result_on_start_again_button_pressed():
	get_tree().reload_current_scene()
	#_result.hide()
	#_result.reload()
	#_startMenu.show()
	#_startMenu.showWelcome(true)
	pass
