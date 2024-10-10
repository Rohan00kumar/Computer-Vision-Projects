import mediapipe as mp
import cv2
import numpy as np
import time

# Constants
ml = 150  # Margin left
max_x, max_y = 350 + ml, 50  # Max dimensions for tool area
curr_tool = "Select Tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0, 0

# Color palette and current color
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
color_idx = 0
current_color = colors[color_idx]

# List to store polygon points
polygon_points = []

# Function to detect which tool is selected


def getTool(x):
    if x < 50 + ml:
        return "line"
    elif x < 100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "circle"
    elif x < 250 + ml:
        return "erase"
    elif x < 300 + ml:
        return "ellipse"
    elif x < 350 + ml:
        return "polygon"
    elif x < 400 + ml:
        return "color"
    else:
        return "none"

# Detect if the index finger is raised


def index_raised(yi, y9):
    return (y9 - yi) > 40


# Mediapipe setup
hands = mp.solutions.hands.Hands(
    min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Load tools image (this image should have all the buttons)
tools_img = cv2.imread("color.png")
tools_img = cv2.resize(tools_img, (max_x - ml, max_y))

# Mask to draw on
mask = np.ones((480, 640)) * 255
mask = mask.astype('uint8')

# Capture from webcam
cap = cv2.VideoCapture(0)
while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)  # Flip image for better user experience
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    op = hands.process(rgb)

    # Check if a hand is detected
    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, mp.solutions.hands.HAND_CONNECTIONS)
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

            # Detect if a tool is selected
            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                # Draw selection indicator
                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("Current tool:", curr_tool)
                    if curr_tool == "color":
                        color_idx = (color_idx + 1) % len(colors)
                        current_color = colors[color_idx]
                    time_init = True
                    rad = 40
            else:
                time_init = True
                rad = 40

            # Drawing tools logic
            if curr_tool == "draw":
                xi, yi = int(i.landmark[12].x *
                             640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)
                if index_raised(yi, y9):
                    cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
                    prevx, prevy = x, y
                else:
                    prevx, prevy = x, y

            elif curr_tool == "line":
                xi, yi = int(i.landmark[12].x *
                             640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)
                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.line(frm, (xii, yii), (x, y), current_color, thick)
                else:
                    if var_inits:
                        cv2.line(mask, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            elif curr_tool == "rectangle":
                xi, yi = int(i.landmark[12].x *
                             640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)
                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.rectangle(frm, (xii, yii), (x, y),
                                  current_color, thick)
                else:
                    if var_inits:
                        cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            elif curr_tool == "circle":
                xi, yi = int(i.landmark[12].x *
                             640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)
                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.circle(frm, (xii, yii), int(
                        ((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), current_color, thick)
                else:
                    if var_inits:
                        cv2.circle(mask, (xii, yii), int(
                            ((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), 0, thick)
                        var_inits = False

            elif curr_tool == "ellipse":
                xi, yi = int(i.landmark[12].x *
                             640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)
                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.ellipse(frm, (xii, yii), (abs(xii - x),
                                abs(yii - y)), 0, 0, 360, current_color, thick)

            elif curr_tool == "polygon":
                xi, yi = int(i.landmark[12].x *
                             640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                # If the index finger is raised, add points to the polygon
                if index_raised(yi, y9):
                    if not var_inits:
                        polygon_points.append((x, y))  # Add point to polygon
                        var_inits = True

                    # Draw temporary lines between the points as the user selects more points
                    if len(polygon_points) > 1:
                        for p in range(len(polygon_points) - 1):
                            cv2.line(
                                frm, polygon_points[p], polygon_points[p+1], current_color, thick)

                else:
                    var_inits = False

                # If 'p' key is pressed, finish the polygon and close it by connecting the last and first point
                if cv2.waitKey(1) & 0xFF == ord('p') and len(polygon_points) > 2:
                    # Draw the polygon on the mask
                    cv2.polylines(mask, [np.array(polygon_points)],
                                  isClosed=True, color=0, thickness=thick)
                    polygon_points.clear()  # Clear points for the next polygon

            elif curr_tool == "erase":
                cv2.circle(frm, (x, y), 30, (0, 0, 0), -1)
                cv2.circle(mask, (x, y), 30, 255, -1)

    # Apply mask and show tool selection
    op = cv2.bitwise_and(frm, frm, mask=mask)
    frm[:, :, 1] = op[:, :, 1]
    frm[:, :, 2] = op[:, :, 2]

    # Draw the tool image
    frm[:max_y, ml:max_x] = cv2.addWeighted(
        tools_img, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

    cv2.putText(frm, curr_tool, (270 + ml, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Paint App", frm)

    # Exit on pressing 'Esc'
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
