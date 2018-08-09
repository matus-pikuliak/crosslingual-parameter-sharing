PYTHON_BIN=/usr/bin/python
if [ -x $PYTHON_BIN ]; then
$PYTHON_BIN test.py setup production $@ >> ~/logfile 2>> ~/logfile
fi
