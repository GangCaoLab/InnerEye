import typing as t

import numpy as np

import bioformats
import javabridge


JVM_STARTED = False


def start_jvm():
    """Start JVM"""
    global JVM_STARTED
    if not JVM_STARTED:
        javabridge.start_vm(class_path=bioformats.JARS)
        JVM_STARTED = True
    # Suppressing JavaExceptions
    # see: https://github.com/LeeKamentsky/python-javabridge/issues/37
    # TODO: suppress error message 'Exception in thread "Thread-0"'
    java_stack = javabridge.make_instance('java/io/ByteArrayOutputStream', "()V")
    java_stack_ps = javabridge.make_instance('java/io/PrintStream',
                                             "(Ljava/io/OutputStream;)V", java_stack)
    javabridge.static_call('Ljava/lang/System;', "setErr",
                           '(Ljava/io/PrintStream;)V', java_stack_ps)
    java_out = javabridge.make_instance('java/io/ByteArrayOutputStream', "()V")
    java_out_ps = javabridge.make_instance('java/io/PrintStream',
                                           "(Ljava/io/OutputStream;)V", java_out)
    javabridge.static_call('Ljava/lang/System;', "setOut",
                           '(Ljava/io/PrintStream;)V', java_out_ps)


def stop_jvm():
    """Stop the JVM.

    NOTE:
    javabridge can not start vm second time...
    JVM need stop before the ending of process.
    """
    javabridge.kill_vm()


def read_series_uri(uri: str) -> np.ndarray:
    return read_series(*parse_uri(uri))


def parse_uri(uri: str) -> t.Tuple[str, str]:
    path, series = uri.split("::")
    return path, series


def read_series(path: str, series: t.Union[str, int]) -> np.ndarray:
    """Read series from a .lif file.

    :param path: Path to the .lif file.
    :param series: series to read.
    :return: Four dimensional array with shape (height, width, depth, channels)
    """
    start_jvm()
    reader = bioformats.ImageReader(path)
    _imgs = []
    z = 0
    while 1:
        try:
            i = reader.read(z=z, series=series)
            _imgs.append(i)
            z += 1
        except javabridge.JavaException:
            break
    arr = np.c_[_imgs]
    arr = arr.swapaxes(0, 2).swapaxes(0, 1)

    return arr

