extends CustomControl

@onready var _menu : Menu = $menu
@onready var _initMenu : InitMenu = $initMenu
@onready var _viewport : ViewportResult = $Viewport
@onready var _warning : CustomWarningMessage = $warning
@onready var _loading : CustomControl = $loading

@onready var _httpResult : HTTPResult = $HTTP/HTTPResult
@onready var _httpImageNames : HTTPImageNames = $HTTP/HTTPImageNames

# Called when the node enters the scene tree for the first time.
func _ready():
	
	Global.httpResult = self._httpResult
	Global.httpImageNames = self._httpImageNames
	
	Global.httpResult.connect("result_loadet", _on_result_loadet)
	
	Global.httpResult.connect("request_failed", _on_request_failed)
	Global.httpImageNames.connect("request_failed", _on_request_failed)
	
	_initMenu.connect("_on_initMenu_closed_with_start", _on_start)
	_initMenu.connect("_on_initMenu_closed_with_load", _on_load)
	
	_initMenu.initalise()
	_loading.initalise()
	
	_menu.connect("_on_new_button_pressed", _on_new_button_pressed)
	_menu.connect("_on_archive_Button_pressed", _on_archive_button_pressed)
	_menu.connect("_on_quit_button_pressed", _on_quit_button_pressed)
	
	pass # Replace with function body.

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func _notification(what):
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		_warning.showWarning("Exit application?" , "Do you really want to exit the application?", "All analyses are automatically stored and archived on the server.", "", false)

func _on_result_loadet(result : AnalysisResult):
	_viewport.setResult(result)
	pass

func _on_start():
	_loading.startLoading()
	pass

func _on_load():
	pass

func _on_new_button_pressed():
	#_startMenu.showWelcome(false)
	
	_initMenu.showMenu("start")
	
	#_httpResult.getResultWithImage(filePath)
	#_httpResult.getResult()
	#_httpImageNames.getAllImageFileNames()
	#_httpResult.getResultWithImageName(_imageNamesList.getSelectedItemText())
	
	#_startMenu.showSelection(true)
	pass

func _on_archive_button_pressed():
	_initMenu.showMenu("archive")
	pass

func _on_request_failed(error : String, info : String):
	
	_warning.showWarning("Request Failed" , error, info, "Please contact the administrator if the warning persists.", false)
	
	pass

func _on_quit_button_pressed():
	
	_warning.showWarning("Exit application?" , "Do you really want to exit the application?", "All analyses are automatically stored and archived on the server.", "", true)
	
	pass
