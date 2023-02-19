/****************************************************************************
** Meta object code from reading C++ file 'main_window.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../soccerwindow2/src/qt4/main_window.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'main_window.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_MainWindow[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      47,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      11,   25,   25,   25, 0x05,

 // slots: signature, parameters, type, tag, flags
      26,   25,   25,   25, 0x08,
      36,   25,   25,   25, 0x08,
      46,   25,   25,   25, 0x08,
      62,   25,   25,   25, 0x08,
      78,   25,   25,   25, 0x08,
      88,   25,   25,   25, 0x08,
     102,   25,   25,   25, 0x08,
     119,   25,   25,   25, 0x08,
     138,   25,   25,   25, 0x08,
     158,   25,   25,   25, 0x08,
     171,   25,   25,   25, 0x08,
     185,   25,   25,   25, 0x08,
     201,  224,   25,   25, 0x08,
     232,  257,   25,   25, 0x08,
     260,   25,   25,   25, 0x08,
     281,  308,   25,   25, 0x08,
     319,   25,   25,   25, 0x08,
     335,   25,   25,   25, 0x08,
     351,   25,   25,   25, 0x08,
     369,   25,   25,   25, 0x08,
     388,   25,   25,   25, 0x08,
     411,   25,   25,   25, 0x08,
     430,  448,   25,   25, 0x08,
     456,   25,   25,   25, 0x08,
     481,   25,   25,   25, 0x08,
     505,   25,   25,   25, 0x08,
     529,   25,   25,   25, 0x08,
     552,   25,   25,   25, 0x08,
     577,  257,   25,   25, 0x08,
     601,   25,   25,   25, 0x08,
     620,   25,   25,   25, 0x08,
     638,   25,   25,   25, 0x08,
     660,   25,   25,   25, 0x08,
     668,   25,   25,   25, 0x08,
     688,  708,   25,   25, 0x08,
     713,   25,   25,   25, 0x08,
     732,   25,   25,   25, 0x0a,
     755,  783,   25,   25, 0x0a,
     789,   25,   25,   25, 0x0a,
     805,  822,   25,   25, 0x0a,
     826,  822,   25,   25, 0x0a,
     847,  822,   25,   25, 0x0a,
     869,  783,   25,   25, 0x0a,
     888,   25,   25,   25, 0x0a,
     902,   25,   25,   25, 0x0a,
     915,   25,   25,   25, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_MainWindow[] = {
    "MainWindow\0viewUpdated()\0\0openRCG()\0"
    "saveRCG()\0openDebugView()\0saveDebugView()\0"
    "kickOff()\0setLiveMode()\0connectMonitor()\0"
    "connectMonitorTo()\0disconnectMonitor()\0"
    "killServer()\0startServer()\0restartServer()\0"
    "restartServer(QString)\0command\0"
    "toggleDragMoveMode(bool)\0on\0"
    "showLauncherDialog()\0changePlayMode(int,QPoint)\0"
    "mode,point\0toggleMenuBar()\0toggleToolBar()\0"
    "toggleStatusBar()\0toggleFullScreen()\0"
    "showPlayerTypeDialog()\0showDetailDialog()\0"
    "changeStyle(bool)\0checked\0"
    "showColorSettingDialog()\0"
    "showFontSettingDialog()\0showMonitorMoveDialog()\0"
    "showViewConfigDialog()\0showDebugMessageWindow()\0"
    "toggleDebugServer(bool)\0startDebugServer()\0"
    "stopDebugServer()\0showImageSaveDialog()\0"
    "about()\0printShortcutKeys()\0"
    "resizeCanvas(QSize)\0size\0saveImageAndQuit()\0"
    "receiveMonitorPacket()\0"
    "updatePositionLabel(QPoint)\0point\0"
    "dropBallThere()\0dropBall(QPoint)\0pos\0"
    "freeKickLeft(QPoint)\0freeKickRight(QPoint)\0"
    "movePlayer(QPoint)\0moveObjects()\0"
    "yellowCard()\0redCard()\0"
};

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        MainWindow *_t = static_cast<MainWindow *>(_o);
        switch (_id) {
        case 0: _t->viewUpdated(); break;
        case 1: _t->openRCG(); break;
        case 2: _t->saveRCG(); break;
        case 3: _t->openDebugView(); break;
        case 4: _t->saveDebugView(); break;
        case 5: _t->kickOff(); break;
        case 6: _t->setLiveMode(); break;
        case 7: _t->connectMonitor(); break;
        case 8: _t->connectMonitorTo(); break;
        case 9: _t->disconnectMonitor(); break;
        case 10: _t->killServer(); break;
        case 11: _t->startServer(); break;
        case 12: _t->restartServer(); break;
        case 13: _t->restartServer((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 14: _t->toggleDragMoveMode((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 15: _t->showLauncherDialog(); break;
        case 16: _t->changePlayMode((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< const QPoint(*)>(_a[2]))); break;
        case 17: _t->toggleMenuBar(); break;
        case 18: _t->toggleToolBar(); break;
        case 19: _t->toggleStatusBar(); break;
        case 20: _t->toggleFullScreen(); break;
        case 21: _t->showPlayerTypeDialog(); break;
        case 22: _t->showDetailDialog(); break;
        case 23: _t->changeStyle((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 24: _t->showColorSettingDialog(); break;
        case 25: _t->showFontSettingDialog(); break;
        case 26: _t->showMonitorMoveDialog(); break;
        case 27: _t->showViewConfigDialog(); break;
        case 28: _t->showDebugMessageWindow(); break;
        case 29: _t->toggleDebugServer((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 30: _t->startDebugServer(); break;
        case 31: _t->stopDebugServer(); break;
        case 32: _t->showImageSaveDialog(); break;
        case 33: _t->about(); break;
        case 34: _t->printShortcutKeys(); break;
        case 35: _t->resizeCanvas((*reinterpret_cast< const QSize(*)>(_a[1]))); break;
        case 36: _t->saveImageAndQuit(); break;
        case 37: _t->receiveMonitorPacket(); break;
        case 38: _t->updatePositionLabel((*reinterpret_cast< const QPoint(*)>(_a[1]))); break;
        case 39: _t->dropBallThere(); break;
        case 40: _t->dropBall((*reinterpret_cast< const QPoint(*)>(_a[1]))); break;
        case 41: _t->freeKickLeft((*reinterpret_cast< const QPoint(*)>(_a[1]))); break;
        case 42: _t->freeKickRight((*reinterpret_cast< const QPoint(*)>(_a[1]))); break;
        case 43: _t->movePlayer((*reinterpret_cast< const QPoint(*)>(_a[1]))); break;
        case 44: _t->moveObjects(); break;
        case 45: _t->yellowCard(); break;
        case 46: _t->redCard(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData MainWindow::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject MainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_MainWindow,
      qt_meta_data_MainWindow, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &MainWindow::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow))
        return static_cast<void*>(const_cast< MainWindow*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 47)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 47;
    }
    return _id;
}

// SIGNAL 0
void MainWindow::viewUpdated()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}
QT_END_MOC_NAMESPACE
