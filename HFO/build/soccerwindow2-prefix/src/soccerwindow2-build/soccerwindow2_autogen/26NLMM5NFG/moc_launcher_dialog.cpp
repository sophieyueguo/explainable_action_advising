/****************************************************************************
** Meta object code from reading C++ file 'launcher_dialog.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../soccerwindow2/src/qt4/launcher_dialog.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'launcher_dialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_LauncherDialog[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      15,   37,   37,   37, 0x05,

 // slots: signature, parameters, type, tag, flags
      38,   37,   37,   37, 0x08,
      52,   37,   37,   37, 0x08,
      68,   37,   37,   37, 0x08,
      85,   37,   37,   37, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_LauncherDialog[] = {
    "LauncherDialog\0launchServer(QString)\0"
    "\0startServer()\0startLeftTeam()\0"
    "startRightTeam()\0startAll()\0"
};

void LauncherDialog::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        LauncherDialog *_t = static_cast<LauncherDialog *>(_o);
        switch (_id) {
        case 0: _t->launchServer((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 1: _t->startServer(); break;
        case 2: _t->startLeftTeam(); break;
        case 3: _t->startRightTeam(); break;
        case 4: _t->startAll(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData LauncherDialog::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject LauncherDialog::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_LauncherDialog,
      qt_meta_data_LauncherDialog, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &LauncherDialog::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *LauncherDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *LauncherDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_LauncherDialog))
        return static_cast<void*>(const_cast< LauncherDialog*>(this));
    return QDialog::qt_metacast(_clname);
}

int LauncherDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void LauncherDialog::launchServer(const QString & _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
