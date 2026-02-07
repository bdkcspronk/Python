import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import time
import math
import random
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Constants
MAX_RIPPLES = 10000
WINDOW_WIDTH = 2560
WINDOW_HEIGHT = 1440
TARGET_FPS = 144
FONT_SIZE = 36
RIPPLE_INTERVAL = 0.01

# Initial shader parameters
OMEGA = 150.0
TAU = 1.5
SMOOTHING = 20.0
AMPLITUDE_SCALE = 0.1

# Color (RGBA)
COLOR_R = 0.2
COLOR_G = 0.5
COLOR_B = 1.0

# Ripple cleanup
AMPLITUDE_THRESHOLD = 0.001  # Remove ripples below this amplitude

# Random sparkles
SPARKLE_INTERVAL = 0.1  # Create random sparkle every 0.1 seconds
SPARKLE_ENABLED = True

# Setup Pygame window
pygame.init()
width, height = WINDOW_WIDTH, WINDOW_HEIGHT
screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)  # Disable vsync
pygame.display.set_caption("GPU Ripple")

# Vertex shader (just pass through)
vertex_shader = """
#version 330
in vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

fragment_shader = """
#version 330
out vec4 fragColor;

uniform vec2 resolution;
uniform float time;
uniform int rippleCount;
uniform vec3 ripples[1000];
uniform float omega;
uniform float tau;
uniform float smoothing;
uniform float amplitude_scale;

void main() {
    vec2 fragPos = gl_FragCoord.xy;

    float alpha = 0.0;
    for(int i = 0; i < rippleCount; i++) {
        vec3 r = ripples[i];
        float t = time - r.z;
        
        if (t >= 0.0) {
            float dist = distance(fragPos, vec2(r.x, r.y));
            float ringRadius = omega * t;
            float distFromRing = abs(dist - ringRadius);
            float ringShape = exp(-distFromRing * distFromRing / (2.0 * smoothing * smoothing));
            float amplitude = amplitude_scale * exp(-tau * t);
            float rippleAlpha = ringShape * amplitude;
            alpha += rippleAlpha;
        }
    }

    fragColor = vec4(0.2, 0.5, 1.0, alpha);
}
"""

# Compile shader program
shader = compileProgram(
    compileShader(vertex_shader, GL_VERTEX_SHADER),
    compileShader(fragment_shader, GL_FRAGMENT_SHADER)
)

# Fullscreen quad
quad = np.array([
    -1, -1,
     1, -1,
    -1,  1,
     1,  1
], dtype=np.float32)

vao = glGenVertexArrays(1)
vbo = glGenBuffers(1)
glBindVertexArray(vao)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
pos = glGetAttribLocation(shader, "position")
glEnableVertexAttribArray(pos)
glVertexAttribPointer(pos, 2, GL_FLOAT, GL_FALSE, 0, None)

# Set up blending for transparency
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE)

# Uniforms
resolution_loc = glGetUniformLocation(shader, "resolution")
time_loc = glGetUniformLocation(shader, "time")
ripple_count_loc = glGetUniformLocation(shader, "rippleCount")
ripples_loc = glGetUniformLocation(shader, "ripples")
omega_loc = glGetUniformLocation(shader, "omega")
tau_loc = glGetUniformLocation(shader, "tau")
smoothing_loc = glGetUniformLocation(shader, "smoothing")
amplitude_scale_loc = glGetUniformLocation(shader, "amplitude_scale")

# Initial values
omega = OMEGA
tau = TAU
smoothing = SMOOTHING
amplitude_scale = AMPLITUDE_SCALE

ripples = []  # [x, y, startTime]
start_time = time.time()
running = True
clock = pygame.time.Clock()

# Setup font for FPS display
font = pygame.font.Font(None, FONT_SIZE)

# Function to add a ripple
def add_ripple(x, y, current_time):
    """Add a ripple at position (x, y) at current_time"""
    ripples.append([float(x), float(y), current_time])
    if len(ripples) > MAX_RIPPLES:
        ripples.pop(0)

def handle_events(current_time):
    """Handle keyboard and mouse input events"""
    global running, omega, tau, smoothing, last_ripple_time
    
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        if event.type == MOUSEBUTTONDOWN:
            x, y = event.pos
            y = height - y
            add_ripple(x, y, current_time)
            last_ripple_time = current_time
        if event.type == KEYDOWN:
            if event.key == K_w:
                omega += 10
            elif event.key == K_s:
                omega = max(10, omega - 10)
            elif event.key == K_a:
                tau += 0.1
            elif event.key == K_d:
                tau = max(0.1, tau - 0.1)
            elif event.key == K_q:
                smoothing += 2
            elif event.key == K_e:
                smoothing = max(2, smoothing - 2)

def handle_mouse_drag(current_time):
    """Handle continuous ripples while dragging mouse"""
    global last_ripple_time
    
    if pygame.mouse.get_pressed()[0]:  # Left mouse button
        if current_time - last_ripple_time > ripple_interval:
            x, y = pygame.mouse.get_pos()
            y = height - y
            add_ripple(x, y, current_time)
            last_ripple_time = current_time

def create_sparkles(current_time):
    """Create random sparkles"""
    global last_sparkle_time
    
    if SPARKLE_ENABLED and current_time - last_sparkle_time > SPARKLE_INTERVAL:
        x = random.randint(0, width)
        y = random.randint(0, height)
        add_ripple(x, y, current_time)
        last_sparkle_time = current_time

def cleanup_ripples(current_time):
    """Remove ripples that have faded below threshold"""
    global ripples
    ripples = [r for r in ripples if AMPLITUDE_SCALE * math.exp(-tau * (current_time - r[2])) > AMPLITUDE_THRESHOLD]

def update_uniforms(current_time):
    """Update all shader uniforms"""
    glUseProgram(shader)
    glUniform2f(resolution_loc, width, height)
    glUniform1f(time_loc, current_time)
    glUniform1i(ripple_count_loc, len(ripples))
    glUniform1f(omega_loc, omega)
    glUniform1f(tau_loc, tau)
    glUniform1f(smoothing_loc, smoothing)
    glUniform1f(amplitude_scale_loc, amplitude_scale)
    
    if len(ripples) > 0:
        ripple_data = np.array(ripples, dtype=np.float32)
        glUniform3fv(ripples_loc, len(ripples), ripple_data)

def render_scene():
    """Render the ripple visualization"""
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
    pygame.display.flip()

# Mouse dragging
last_ripple_time = 0
ripple_interval = RIPPLE_INTERVAL

# Sparkles
last_sparkle_time = 0

while running:
    dt = clock.tick(TARGET_FPS) / 1000
    current_time = time.time() - start_time
    fps = clock.get_fps()
    
    # Handle input and updates
    handle_events(current_time)
    handle_mouse_drag(current_time)
    create_sparkles(current_time)
    cleanup_ripples(current_time)
    
    # Render
    update_uniforms(current_time)
    render_scene()
    
    # Update window title
    pygame.display.set_caption(f"GPU Ripple - FPS: {fps:.1f} | omega: {omega:.1f} tau: {tau:.2f} smooth: {smoothing:.1f}")

pygame.quit()