#!/usr/bin/env python3
import base64
import concurrent
import glob
import itertools
import math
import os
import re
import time
from timeit import default_timer as timer

import cv2
import numpy as np
import pyautogui
import pyperclip
import pytesseract
import win32gui
from mss import mss
from pynput import keyboard

types = []
sct = None
tesseractfont = None


def tessdata(img):
    global tesseractfont
    return pytesseract.image_to_data(
        img,
        output_type='data.frame',
        lang=tesseractfont,
        config=r'--psm 11 -c load_system_dawg=0')


def myround(x, base=45):
    return int(round(x / base)) * base


def typeEval(param):
    """Find the type icons in the image"""
    matches = np.vstack(
        np.where(
            cv2.matchTemplate(param[0], param[1][1], cv2.TM_CCOEFF_NORMED) >= 0.8
        )
    ).T
    if matches.any():
        return matches[0][0], matches[0][1], param[1][0]


def itemEval(base, debug=False, wdebug=False):
    """Take the screenshot of the item and filter it to pass to tesseract"""
    # need base for colour matching and img for tesseract
    base = cv2.cvtColor(np.array(base), cv2.IMREAD_COLOR)

    img = cv2.cvtColor(np.array(base), cv2.IMREAD_COLOR)
    # remove everything < 170 or 180 to clean it up a little do not use 0, 0, 0 or it will go white in the next step
    img[np.where((img < [173, 173, 173]).all(axis=2))] = [20, 0, 0]

    # make equal colours white, 190, 190 ,190 -> 255, 255, 255
    img[np.where((img[:, :, 0] == img[:, :, 1]) & (img[:, :, 1] == img[:, :, 2]))] = [255, 255, 255]

    img[np.where((img < [255, 255, 255]).any(axis=2))] = [0, 0, 0]

    # call tesseract on the image, inverse reads better
    data = tessdata(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)[1])

    data = data[data.text.notnull()]

    # wdebug before stuff is removed
    if wdebug:
        # put all the tesseract text on the image and save it
        [cv2.putText(img, f'{text}', (l, t), cv2.QT_FONT_NORMAL, 0.55, (0, 255, 0))
         for text, t, l in zip(data.text, data.top, data.left)]
        cv2.imwrite(f'debug{time.time()}.png', img)

    if debug:
        cv2.imshow(f'TessImage', img)
        cv2.waitKey(0)

    # filter out random image text
    data = data[data.text.map(len) < 6]
    # remove ilvl
    data = data[~data.text.isin(['083', '084', '81', '82', '83', '84', '85', '86'])]

    data.text = data.text.apply(
        lambda row: [int(i) for i in filter(None, row.split('/'))] if '/' in row else [int(row)]
    )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        global types
        matches = [ret.result() for ret in [executor.submit(typeEval, (base, ty)) for ty in types]]

        for y, x, ty in [match for match in matches if match is not None]:
            # match a type to some text on the left of it
            for row in data[['left', 'top']].itertuples():
                # types are always to the right and should be almost level with the text
                if x < row.left or myround(y) != myround(row.top):
                    continue

                # match close types to the text
                if (x - row.left) ** 2 + (y - row.top) ** 2 < 1520:
                    data.text.loc[row.Index].insert(0, ty)

        return data.text.tolist()


def click():
    time.sleep(0.01)
    pyautogui.click()


def moveMouse(topl, slot):
    # Move mouse to the slot based on the top left location
    wh = 53  # size of a inventory box
    x = 3 + topl[0] + slot[0] * wh
    y = 3 + topl[1] + slot[1] * wh
    pyautogui.moveTo(x, y)


def testBP():
    """Screen shot and parse ctrl c data"""
    global sct
    moveMouse((325, 205), (0, 0))

    # screen shot the item
    # time.sleep(0.03)
    pyautogui.keyDown('alt', _pause=False)
    # need these sleeps as it takes a small amount of time for alt to show up
    time.sleep(0.05)

    im = np.array(
        sct.grab({
            'left': 200,
            'top': 240,
            'width': 1050,
            'height': 260
        }))

    time.sleep(0.02)
    pyautogui.keyUp('alt')

    pyautogui.hotkey('ctrl', 'c')
    item = pyperclip.paste()

    for _ in range(10):
        if re.search('Wings Revealed: (.*?)\r\n', item) is None:
            moveMouse((325, 205), (0.3, 0.3))
            moveMouse((325, 205), (0, 0))
            pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.05)
            item = pyperclip.paste()
        else:
            break
    else:
        exit('No item in slot or cannot ctrl-c')

    return itemEval(im), base64.b64encode(item.encode("utf-8"))


def putInTrade(comb, intrade):
    """Put a given combination in the trade window"""
    # work out what slots the combination needs 6 -> 1, 1
    slots = [(math.floor(x / 5), (x % 5)) for x in comb]

    removal = []
    for item in intrade:
        if item not in slots:
            moveMouse((312, 535), item)
            pyautogui.keyDown('ctrl')
            click()
            pyautogui.keyUp('ctrl')

            removal.append(item)

    for x in removal:
        intrade.remove(x)

    for slot in slots:
        if slot not in intrade:
            moveMouse((1275, 590), slot)
            click()

            moveMouse((312, 535), slot)
            click()

            intrade.append(slot)

    return testBP()


def pComb(n, r):
    return int(math.factorial(n) / (math.factorial(r) * math.factorial(n - r)))


def bruteForceBP(start_at=(0, 1, 2, 3, 4), used_items=[], lname='log.txt', searchfor=[]):
    print(f'Combinations: {pComb(60 - len(used_items), 5)}')
    with open(lname, 'a') as log:

        count = 0
        stop = False
        # want to keep track of what we currently have in the trade window
        intrade = []
        for combination in itertools.combinations(range(60), 5):
            # skip to the combination to start at
            if combination == start_at:
                stop = True
            else:
                if not stop:
                    continue

            start = timer()
            # check if a element has been used in the set
            if any(c in combination for c in used_items):
                continue

            values, text = putInTrade(combination, intrade)

            # if value is high enough dont let the items be used again
            for finding in values:
                if any(v in finding for v in searchfor):
                    used_items.extend(combination)
                    print(f'Used: {combination}')

            log.write(f'{combination}\t{values}\t{text}\n')
            log.flush()

            print(f'Perm: {count}\t{combination}\tTime:{timer() - start}')
            count = count + 1

    print(f'Finished: {used_items}')


def on_press(key):
    try:
        k = key.char
    except:
        if key.name == 'esc':
            # input('Press key to resume')
            print(f'ExitFromEsc')
            os._exit(1)


def main():
    time.sleep(1)
    # set up exit as mouse will be unusable and console out of focus
    lis = keyboard.Listener(on_press=on_press)
    lis.start()

    # set up screen shot taker
    global sct
    sct = mss()

    global tesseractfont
    # font created from Tesseract-OCR data folder
    tesseractfont = 'poe'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    win = win32gui.FindWindow(None, 'Path Of Exile')
    # bring path to the front
    if win == 0:
        exit('No Game')

    win32gui.ShowWindow(win, 5)
    win32gui.SetForegroundWindow(win)

    # set the default pause value from 0.1 to something a little faster
    pyautogui.PAUSE = 0.01

    global types
    for filepath in glob.iglob('types/' + r'*'):
        print(f'Loaded: {filepath}')
        types.append([
            os.path.splitext(
                os.path.basename(filepath))[0],
            cv2.imread(filepath)])

    test = [
        # (6, 7, 9, 16, 17)
    ]

    if test:
        [print(putInTrade(x, [])) for x in test]
        exit(0)

    # if inventory slots are empty add them here
    used_items = []

    bruteForceBP(used_items=used_items)


if __name__ == '__main__':
    main()
