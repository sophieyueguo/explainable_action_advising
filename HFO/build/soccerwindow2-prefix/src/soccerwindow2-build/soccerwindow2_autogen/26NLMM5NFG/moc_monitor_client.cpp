/****************************************************************************
** Meta object code from reading C++ file 'monitor_client.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../soccerwindow2/src/qt4/monitor_client.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'monitor_client.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_MonitorClient[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: signature, parameters, type, tag, flags
      14,   25,   25,   25, 0x05,
      26,   25,   25,   25, 0x05,

 // slots: signature, parameters, type, tag, flags
      36,   25,   25,   25, 0x08,
      52,   25,   25,   25, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_MonitorClient[] = {
    "MonitorClient\0received()\0\0timeout()\0"
    "handleReceive()\0handleTimer()\0"
};

void MonitorClient::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        MonitorClient *_t = static_cast<MonitorClient *>(_o);
        switch (_id) {
        case 0: _t->received(); break;
        case 1: _t->timeout(); break;
        case 2: _t->handleReceive(); break;
        case 3: _t->handleTimer(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData MonitorClient::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject MonitorClient::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_MonitorClient,
      qt_meta_data_MonitorClient, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &MonitorClient::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *MonitorClient::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *MonitorClient::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_MonitorClient))
        return static_cast<void*>(const_cast< MonitorClient*>(this));
    return QObject::qt_metacast(_clname);
}

int MonitorClient::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    }
    return _id;
}

// SIGNAL 0
void MonitorClient::received()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}

// SIGNAL 1
void MonitorClient::timeout()
{
    QMetaObject::activate(this, &staticMetaObject, 1, 0);
}
QT_END_MOC_NAMESPACE
