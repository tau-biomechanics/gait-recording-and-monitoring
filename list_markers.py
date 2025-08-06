import opensim as osim

# Load the model
model_path = 'data/opencap_data/OpenCapData_eec9c938-f468-478d-aab1-4ffcb0963207_1507-2/OpenSimData/Model/LaiUhlrich2022_scaled.osim'
model = osim.Model(model_path)

# Get marker set
marker_set = model.getMarkerSet()
print(f'Model has {marker_set.getSize()} markers:')

# Print all marker names
for i in range(marker_set.getSize()):
    print(f'  {i+1}. {marker_set.get(i).getName()}') 