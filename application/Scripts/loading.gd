extends CustomControl

@onready var _progressBar : ProgressBar = $Panel/ProgressBar
@onready var _timer : Timer = $Timer
@onready var _closeButton : Button = $Panel2/closeButton
@onready var _label : Label = $Panel/Label

var _currentValue = 0
var _hide = false
var _nextIdle = false

signal loading_finished
signal loading_next

# Called when the node enters the scene tree for the first time.
func _ready():	
	pass # Replace with function body.

func initalise():
	_timer.wait_time = 2
	_timer.connect("timeout", _on_Timer_timeout)
	
	Global.httpImageNames.connect("current_server_state", _on_state_update)
	
	_closeButton.connect("pressed", _close_Button_pressed)
	pass

func _close_Button_pressed():
	_timer.stop()
	_progressBar.value = 0
	self.hide()
	_hide = false
	_nextIdle = false
	pass

func _on_Timer_timeout():
	# Code, der alle 5 Sekunden ausgeführt wird
	#counter += 1
	#print("Code wird alle 5 Sekunden ausgeführt. Counter:", counter)
	
	# Überprüfen, ob es nicht mehr nötig ist, den Code auszuführen
	#if counter >= 10:
	#	$Timer.stop()
	#	print("Code wird nicht mehr benötigt.")
	Global.httpImageNames.getProcessUpdate()
	pass

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func startLoading():
	self.show()
	Global.httpImageNames.getProcessUpdate()
	_timer.start()
	pass

func next():
	var tween = _progressBar.create_tween()
	tween.tween_property(_progressBar, "modulate", Color(1,1,1,0), 2).from(Color(1,1,1,1)).set_trans(Tween.TRANS_CUBIC)
	tween.tween_callback(func(): loading_next.emit())

func loadNextSection(start : int, end : int):
	var tween = _progressBar.create_tween()
	tween.tween_property(_progressBar, "value", end, 7).from(start).set_trans(Tween.TRANS_LINEAR)
	#tween.tween_callback(func(): loading_finished.emit())
	pass

func _on_state_update(state : String):
	
	if _hide or self.visible == false:
		_timer.stop()
		_progressBar.value = 0
		self.hide()
		_hide = false
		_nextIdle = false
		return
	
	var multiplier = 0
	
	match state:
		"load_image":
			multiplier = 1
			pass
		"preprocess_image":
			multiplier = 2
			pass
		"prediction":
			multiplier = 3
			_nextIdle = true
			pass
		"interpretability":
			multiplier = 4
			_nextIdle = true
			pass
		"convert_visualization":
			multiplier = 5
			pass
		"convert_batch":
			multiplier = 6
			pass
		"convert_lime":
			multiplier = 7
			pass
		"convert_original":
			multiplier = 8
			pass
		"create_archive_folder":
			multiplier = 9
			pass
		"archive_original":
			multiplier = 10
			pass
		"archive_visualization":
			multiplier = 11
			pass
		"archive_batch":
			multiplier = 12
			pass
		"archive_lime":
			multiplier = 13
			pass
		"archive_scores":
			multiplier = 14
			pass
		"archive_scores_road":
			multiplier = 15
			pass
		"archive_predictions":
			multiplier = 16
		"finish":
			multiplier = 17
			_hide = true
			_nextIdle = true
			pass
		"idle":
			if _nextIdle:
				_timer.stop()
				_progressBar.value = 0
				self.hide()
				return
	
	_label.text = state
	
	_progressBar.value = 5.882352941176471 * multiplier
	
	pass
