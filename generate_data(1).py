import carla
import os
import time
import random
import csv

# --- KONFIGURACJA ---
WIDTH = 2896
HEIGHT = 1876
FOV = 120
BASELINE = 0.2  # 20 cm
OUTPUT_DIR = "dataset_stereo"
NUM_FRAMES = 1000

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0) 
    
    world = client.load_world('Town03')
    
    vehicle = None
    cam_l = None
    cam_r = None
    npc_list = []

    print("Warming up renderer...")
    for _ in range(50):  # ~50 ticks warm-up
        world.tick()
    try:

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.025
        world.apply_settings(settings)

        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)

        blueprint_library = world.get_blueprint_library()
        
        bp = blueprint_library.filter('model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(bp, spawn_point)
        vehicle.set_autopilot(True, 8000)

        # 5. Ożywianie miasta - spawnowanie 30 aut NPC
        print("Spawnowanie ruchu ulicznego...")
        v_blueprints = blueprint_library.filter('vehicle.*')
        for _ in range(30):
            v_bp = random.choice(v_blueprints)
            v_spawn = random.choice(spawn_points)
            npc = world.try_spawn_actor(v_bp, v_spawn)
            if npc:
                npc.set_autopilot(True, 8000)
                npc_list.append(npc)

        # 6. Konfiguracja kamer Stereo (IMX490)
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(WIDTH))
        cam_bp.set_attribute('image_size_y', str(HEIGHT))
        cam_bp.set_attribute('fov', str(FOV))

        spawn_l = carla.Transform(carla.Location(x=1.6, z=1.2, y=-BASELINE/2))
        spawn_r = carla.Transform(carla.Location(x=1.6, z=1.2, y=BASELINE/2))

        cam_l = world.spawn_actor(cam_bp, spawn_l, attach_to=vehicle)
        cam_r = world.spawn_actor(cam_bp, spawn_r, attach_to=vehicle)

        os.makedirs(f"{OUTPUT_DIR}/left", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/right", exist_ok=True)
        
        csv_file = open(f"{OUTPUT_DIR}/ground_truth.csv", mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'class_name', 'object_id', 'true_distance_m'])

        cam_l.listen(lambda image: image.save_to_disk(f"{OUTPUT_DIR}/left/{image.frame:06d}.jpg"))
        cam_r.listen(lambda image: image.save_to_disk(f"{OUTPUT_DIR}/right/{image.frame:06d}.jpg"))

        print(f"Rozpoczynam zbieranie {NUM_FRAMES} klatek...")
        for i in range(NUM_FRAMES + 1):
            world.tick()
            
            v_loc = vehicle.get_location()
            
            vehicles = world.get_actors().filter('vehicle.*')
            pedestrians = world.get_actors().filter('walker.pedestrian.*')
            
            targets = list(vehicles) + list(pedestrians)
            
            for target in targets:
                if target.id == vehicle.id:
                    continue
                
                t_loc = target.get_location()
                dist = v_loc.distance(t_loc)
                
                if dist < 50.0:
                    csv_writer.writerow([i, target.type_id, target.id, round(dist, 3)])

            if i % 10 == 0:
                print(f"Postęp: {i}/{NUM_FRAMES} klatek")

    except Exception as e:
        print(f"BŁĄD: {e}")

    finally:
        print("Sprzątanie i zamykanie plików...")
        if 'csv_file' in locals():
            csv_file.close()
            
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        if cam_l: cam_l.destroy()
        if cam_r: cam_r.destroy()
        if vehicle: vehicle.destroy()
        for npc in npc_list:
            npc.destroy()

if __name__ == '__main__':
    main()
