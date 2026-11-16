<p align="center">
  <img src="assets/hero.svg" width="100%" alt="A ray-marched room at night, drawn in ASCII. A red neon sign on the back wall spells ARDA ÇABUK and stutters now and then. In the foreground someone with short curly dark-blond hair and headphones sits in profile at a desk against the left wall, lit by an amber lamp, nodding to the music and typing; the monitor reads arda@jetson and shows a scrolling vibration trace marked nominal, a development board blinks and a pair of drumsticks lies by the keyboard. A drum kit stands in the moonlight, and through a tall arched window a cat swishes its tail on the sill against the horizon under a crescent moon, stars twinkle, shooting stars cross, cars pass on the hill road and wind turbines turn on the ridge with red lights blinking.">
</p>

### Hi, I'm Arda.

I write software for things you can put a probe on. Some days that is C written against the reference manual, one register at a time, on a Cortex-M4. Other days it is a handful of C++ processes passing messages on a Raspberry Pi, or a Jetson turning 28 channels of vibration into spectra. Once the data is clean enough to learn from, I switch to Python and PyTorch.

The two halves meet more often than you would expect, because most of what I build ends up answering one question: **is this still normal?** A switch that bounces. A turbine that starts humming at the wrong frequency. A crack in a board, a leaf that is sick before it looks sick. The picture above is a night like that: turbines turning on the ridge, their vibration on the screen, and someone checking.

I graduated from Eskişehir Osmangazi University in 2026 with a B.Sc. in Computer Engineering and a 3.83 GPA, first in my department and in the faculty. I'm looking for a role in **embedded software** or **machine learning**, ideally somewhere the two touch.

### From the register up

<img src="assets/stm32.svg" width="100%" alt="A logic-analyser view of one key press on a 4x4 keypad, its traces glowing brightest where they are written and fading as they scroll. The scan column steps across the keypad while the switch contact chatters; the pressed key sinks into the board. The debounce state machine drops back to IDLE when a bounce drops out, reaches PRESSED once the contact has been stable for 20 ms, and returns the key exactly once while it stays held.">

**[stm32f411ceux_driver](https://github.com/ardacbk/stm32f411ceux_driver)** · C · ARM Cortex-M4 · no HAL<br>
RCC, GPIO, SysTick and EXTI/NVIC drivers for the STM32F411 "Black Pill", written from the reference manual with no HAL or LL underneath. A 4×4 keypad BSP sits on top of the GPIO driver: it drives one column at a time, reads the rows through pull-downs, and runs a three-state machine (`IDLE`, `DEBOUNCE`, `PRESSED`) on SysTick timestamps, so a key that is held down is reported exactly once. The trace above is that state machine working through a contact that bounces. It builds with CMake, Ninja and arm-none-eabi-gcc.

<img src="assets/raspi.svg" width="100%" alt="Four processes drawn as boxes: ClientNode on the PC, ServerNode as a router on the Raspberry Pi, DigitalIONode with a sensor, a relay and a keypad, and EnvSensorNode with its UART loop. Commands such as relay 1 and temp ease along the wires as text with a fading trail, lighting the wire behind them, are routed by their prefix, and the replies travel back to the client's terminal. The relay flips in red.">

**[raspi-remote-control-system](https://github.com/ardacbk/raspi-remote-control-system)** · C++17 · Linux · Raspberry Pi 3<br>
Four processes split across a PC and a Raspberry Pi. The client talks TCP to a router process, which looks at the start of each message and forwards it to the process that owns the hardware. One of them drives a sensor, a relay and a 4×4 keypad over GPIO with libgpiod. The other answers temperature and humidity queries from a second thread sitting on the UART at `/dev/serial0`.

<img src="assets/turbine.svg" width="100%" alt="A signal pipeline read left to right: scrolling vibration traces, their FFT spectrum, a spectrogram drawn as a dense heat map, and the reconstruction error of an autoencoder against a threshold. Once per loop a fault appears in the signal; a new band shows up in the spectrum and the error crosses the threshold in red.">

**Graduation project: turbine condition monitoring** · NVIDIA Jetson · PyTorch · MQTT · TimescaleDB<br>
A Jetson reads 28 vibration channels and turns them into 426-dimensional FFT feature vectors and spectrograms. Two kinds of model then decide whether the machine still sounds like itself: LSTM autoencoders watch the reconstruction error, and a CNN reads the spectrograms. Everything streams over MQTT (TLS, MessagePack) into TimescaleDB hypertables.

### Models that notice

<table>
<tr>
<td width="50%" valign="top">
<img src="assets/wood.svg" width="100%" alt="A wooden board printed from a rendered image, its grain bending around a dark knot, with a thin crack. A scanner line sweeps it three times, once each for GLASS, EfficientAD and INP-Former, and paints that model's anomaly map in red: strong on the crack, quiet on the knot.">
<p><b><a href="https://github.com/ardacbk/wood_anomaly_detection">wood_anomaly_detection</a></b> · PyTorch · Flask · React<br>
Three pretrained anomaly detectors, GLASS, EfficientAD and INP-Former, look at the same wooden board. A Flask backend runs them and a React frontend puts their anomaly maps side by side, so you can see where they agree on a crack and where one of them is fooled by a knot.</p>
</td>
<td width="50%" valign="top">
<img src="assets/leaves.svg" width="100%" alt="A shaded ASCII leaf with bright veins next to two sensor traces and a disease-probability meter. The meter crosses one half and turns red while the leaf still looks healthy; only afterwards do red lesions spread across the blade.">
<p><b>TÜBİTAK-funded research</b> · 2024 – 2025<br>
Models that catch plant diseases early, from two kinds of evidence at once: images of the leaves and readings from sensors around them. I also built the research group's website with Spring Boot.</p>
</td>
</tr>
<tr>
<td width="50%" valign="top">
<img src="assets/ckd.svg" width="100%" alt="A table of illustrative kidney-disease patients with about a quarter of the values missing, shown as question marks. A cursor reads the rows one by one: present values light up, each missing value becomes a green mask instead of being filled in, and a probability of chronic kidney disease appears with a bar, in red when it is above one half. Under the table, bars show each feature's evidence for the row being read: up towards disease, down towards healthy, nothing for a masked feature.">
<p><b><a href="https://github.com/ardacbk/NAIM-CKD">NAIM-CKD</a></b> · PyTorch · scikit-learn<br>
Clinical tables are full of holes. NAIM is a transformer that masks missing values in attention instead of inventing numbers for them. I applied it to the UCI Chronic Kidney Disease dataset and compared it with SVM, KNN and Random Forest trained on median/mode-imputed data.</p>
</td>
<td width="50%" valign="top">
<img src="assets/alignment.svg" width="100%" alt="A solid, lit plane learns to align with a dashed green target plane; its shading changes with its tilt. Three rollouts from the same tilted pose, taken early, midway and late in training: the first wobbles, the second overshoots, the third settles and flashes red. A learning curve sits on the right.">
<p><b>ASELSAN internship</b> · 2025 · Gymnasium<br>
In Microelectronics and Electro-Optical Systems I modelled a 3D-plane alignment task as a Gymnasium environment and trained a reinforcement learning agent to do the alignment on its own.</p>
</td>
</tr>
</table>

### For the fun of it

<table>
<tr>
<td width="50%" valign="top">
<img src="assets/sand.svg" width="100%" alt="A falling-sand automaton in ASCII: water pours into a stone basin in falling streams and darkens with depth, sand piles up on a ledge, slides over the edge and sinks through the water until the basin overflows. Then the floor opens and everything drains away.">
<p><b><a href="https://github.com/ardacbk/sand-simulation">sand-simulation</a></b> · C++ · SFML<br>
A falling-sand cellular automaton. Every cell holds an <code>Element</code>, and Sand, Water and Stone each bring their own <code>update()</code>. Sand falls, slides diagonally and sinks through water. Water flows down, then sideways. Stone stays put. The picture runs the same rules.</p>
</td>
<td width="50%" valign="top">
<img src="assets/raytracer.svg" width="100%" alt="An ASCII path tracer: a glossy sphere, a matte one and a glowing red one hovering above a dark mirror floor start as noisy speckle and settle into shaded spheres with their reflections below them as samples accumulate. The camera glides to a new view, the picture turns to noise while it moves, and accumulation starts over.">
<p><b><a href="https://github.com/ardacbk/RayTracing">RayTracing</a></b> · C++ · Walnut<br>
A ray tracer with reflective and emissive materials and progressive accumulation, built along The Cherno's series. Every frame adds one noisy sample per pixel, and the image settles as the average grows. Move the camera and it starts over.</p>
</td>
</tr>
<tr>
<td width="50%" valign="top">
<img src="assets/particles.svg" width="100%" alt="An ASCII Verlet simulation. A nozzle at the upper left of a round container fires a dense stream of particles that streak across, bounce off the wall, slide down and pile up into a heap that fills the lower half; their tone shows their speed and the newest are red. Then the bottom opens, the heap pours out in falling streaks, and the empty container is ready for the next run.">
<p><b><a href="https://github.com/ardacbk/particle-physics-simulation">particle-physics-simulation</a></b> · C++ · SFML<br>
Verlet-integrated particles with gravity, collisions between every pair and a boundary constraint that keeps them in the bowl.</p>
</td>
<td width="50%" valign="top">
<p><b>Also on the shelf</b></p>
<ul>
<li><a href="https://github.com/ardacbk/water-sorting-game"><code>water-sorting-game</code></a>: the water-sort puzzle, in C++</li>
<li><a href="https://github.com/ardacbk/leetcode-solutions"><code>leetcode-solutions</code></a>: what it says, mostly C++</li>
<li><a href="https://github.com/ardacbk/tcdd-empty-seat-finder"><code>tcdd-empty-seat-finder</code></a>: a Spring Boot API for anyone who has refreshed the TCDD ticket page one too many times</li>
</ul>
</td>
</tr>
</table>

### Along the way

| When | Where | What |
| --- | --- | --- |
| 2026 | **Alp Havacılık** · IT intern | A desktop scheduler in C#. It works out start and end times for jobs and sequences them across the available resources, with SQL Server underneath and a WinForms/WPF interface on top. |
| 2025 | **ASELSAN** · summer intern | The plane-alignment agent above. |
| 2024 – 25 | **TÜBİTAK** · volunteer researcher | The plant-disease models above. |
| 2024 – 25 | **TÜSEB** · scholarship researcher | A Unity VR simulation that teaches elderly people how to use an ATM. |
| 2022 – 26 | **Eskişehir Osmangazi University** | B.Sc. Computer Engineering, 3.83 / 4.00, first in the department and in the Faculty of Engineering and Architecture. |

### Toolbox

| Layer | What I reach for |
| --- | --- |
| **Metal** | C · STM32 / ARM Cortex-M4 · CMSIS · register-level drivers · GPIO · EXTI/NVIC · SysTick · arm-none-eabi-gcc · ST-Link |
| **Systems** | C++17 · STL · POSIX threads · TCP/IP sockets · UART · libgpiod · Raspberry Pi · NVIDIA Jetson · Linux · CMake · Docker |
| **Signals and data** | Python · NumPy · pandas · FFT · spectrograms · MQTT (TLS) · MessagePack · TimescaleDB · PostgreSQL · SQL Server |
| **Models** | PyTorch · TensorFlow · scikit-learn · CNNs · LSTM autoencoders · reinforcement learning (Gymnasium) · anomaly detection · LLM APIs · embeddings · RAG |

I also work in C#, Java, Spring Boot, React, .NET and Unity. Turkish is my native language and my English is C1 (YDS 85/100).

### Say hello

[LinkedIn](https://www.linkedin.com/in/arda-%C3%A7abuk-8355b2253) · [ardacabuk1@hotmail.com](mailto:ardacabuk1@hotmail.com) · [LeetCode](https://leetcode.com/u/prathos/)

<p align="center"><sub>Every picture on this page is computed. Each one is a small simulation (a ray-marched room, a debounce state machine, a path tracer, a cellular automaton) drawn as characters by the scripts in <a href="assets/src">assets/src</a> and shipped as a plain SVG: no JavaScript, no GIFs, light and dark.</sub></p>
