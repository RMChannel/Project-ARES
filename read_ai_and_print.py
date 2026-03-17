import turtle

# Import the extraction function from your first file 
# (Make sure the first file is saved as parser.py)
import read_ai

def draw_circuit(points):
    if not points:
        print("No points found to draw.")
        return

    # 1. Setup the Turtle screen
    screen = turtle.Screen()
    screen.title("Circuit Test Viewer")
    screen.bgcolor("black")

    # 2. Automatically scale the screen to fit the coordinates
    # Note: If the circuit looks like a flat line, the top-down view in your game 
    # might use X and Z instead of X and Y. If so, change p.y to p.z below!
    min_x = min(p.x for p in points)
    max_x = max(p.x for p in points)
    min_y = min(p.y for p in points)
    max_y = max(p.y for p in points)

    # Add a 10% padding so the circuit doesn't touch the window borders
    pad_x = (max_x - min_x) * 0.1
    pad_y = (max_y - min_y) * 0.1
    
    # This is the magic command that forces the window to match your custom coordinates
    screen.setworldcoordinates(min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y)

    # 3. Setup the Turtle pen
    pen = turtle.Turtle()
    pen.speed("fastest") # Maximum drawing speed
    pen.color("cyan")
    pen.pensize(2)
    pen.hideturtle()     # Hide the arrow icon for a cleaner look

    # 4. Start drawing
    pen.penup()
    # Move to the very first point
    pen.goto(points[0].x, points[0].y)
    pen.pendown()

    # Loop through the rest of the points and draw lines between them
    for p in points[1:]:
        pen.goto(p.x, p.y)

    print(f"Finished drawing {len(points)} points!")
    
    # Keep the window open until the user clicks on it
    screen.exitonclick()

# --- EXECUTION ---
if __name__ == "__main__":

    # Load the points using the function from your first file
    # Make sure "fast_lane.ai" is in the same folder
    lista_punti = read_ai.get_data("files_ai/fast_lane.ai")
    
    # Start the drawing process
    draw_circuit(lista_punti)