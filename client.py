#! /usr/bin/python3

import getopt
import os
import select
import socket
import subprocess
import sys
import time
import threading
import traceback

def usage():
    print('-e x  program to invoke (Ataxx "engine")')
    print('-i x  server to connect to')
    print('-p x  port to connect to (usually 28028)')
    print('-U x  username to use')
    print('-P x  password to use')

engine = None
host = 'server.ataxx.org'
port = 28028
user = None
password = None

try:
    optlist, args = getopt.getopt(sys.argv[1:], 'e:i:p:U:P:')

    for o, a in optlist:
        if o == '-e':
            engine = a
        elif o == '-i':
            host = a
        elif o == '-p':
            port = int(a)
        elif o == '-U':
            user = a
        elif o == '-P':
            password = a
        else:
            print(o, a)

except getopt.GetoptError as err:
    print(err)
    usage()
    sys.exit(1)

if user == None or password == None:
    print('No user or password given')
    usage()
    sys.exit(1)

def engine_thread(sck, eng):
    try:
        while True:
            dat = eng.stdout.readline()
            if dat == None:
                break

            dat = dat.replace(b'\n', b'\r\n')
            print(time.asctime(), 'engine: ', dat)

            rc = sck.send(dat)
            print('engine rc: ', rc)
            if rc == 0:
                break

    finally:
        print('Terminating engine_thread: close process')
        eng.kill()

        print('Terminating engine_thread: close socket')
        sck.close()

        print('Engine_thread terminated')

def socket_thread(eng, sck):
    try:
        while True:
            dat = sck.recv(4096)
            if dat == None:
                break

            print(time.asctime(), 'socket: %s' % dat.decode())
            rc = eng.stdin.write(dat)
            print('socket rc: ', rc)
            if rc == 0:
                break

            eng.stdin.flush()

    finally:
        print('Terminating socket_thread: close socket')
        sck.close()

        print('Terminating socket_thread: close process')
        eng.kill()

        print('Socket_thread terminated')

while True:
    try:
        print('Start process')
        p = subprocess.Popen(engine, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))

        s.send(bytes('user %s\n' % user, encoding='utf8'))
        s.send(bytes('pass %s\n' % password, encoding='utf8'))

        t1 = threading.Thread(target=socket_thread, args=(p, s, ))
        t1.start()

        t2 = threading.Thread(target=engine_thread, args=(s, p, ))
        t2.start()

        t2.join()
        print('Back from engine_thread join')

        t1.join()
        print('Back from socket_thread join')

    except ConnectionRefusedError as e:
        print('failure: %s' % e)
        time.sleep(2.5)

    except Exception as e:
        print('failure: %s' % e)
        traceback.print_exc(file=sys.stdout)
        break

    finally:
        print('Close socket')
        s.close()
        del s

        print('Terminate process')
        p.stdout.close()
        p.stdin.close()
        p.kill()
        p.wait()
        del p
