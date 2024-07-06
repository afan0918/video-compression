from visualize_motion import *

reference_image_path = 'img/foreman_qcif_0_rgb.bmp'
current_image_path = 'img/foreman_qcif_1_rgb.bmp'
reference_image = load_image(reference_image_path)
current_image = load_image(current_image_path)

reference_luma = extract_luma(reference_image)
current_luma = extract_luma(current_image)

# Full Search Motion Estimation
mv_full = full_search(reference_luma, current_luma)
visualize_motion_vectors(mv_full)

mv_spiral = full_search_spiral(reference_luma, current_luma)
visualize_motion_vectors(mv_spiral)

# Diamond Search Motion Estimation
mv_diamond = diamond_search(reference_luma, current_luma)
visualize_motion_vectors(mv_diamond)

# Intra Frame Prediction
predicted_frame, modes = intra_prediction(reference_luma)
visualize_modes(modes)
print(modes)
