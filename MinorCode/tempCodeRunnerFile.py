from pathlib import Path
import pandas as pd # type: ignore

# ---------------------------------------------------------
# PART 1: EXTRACT FOLDER PATHS
# ---------------------------------------------------------
root_dir = Path(r"D:\ADNIzip1\ADNI")
data = []

print("Scanning folders for DICOM images...")
for dcm_file in root_dir.rglob("*.dcm"):
    folder_path = dcm_file.parent
    parts = folder_path.parts
    
    image_id = parts[-1]                 
    date_folder = parts[-2]              
    modality = parts[-3]                 
    ptid = parts[-4]                     
    
    scan_date = date_folder.split('_')[0]
    
    scan_info = {
        "PTID": ptid,
        "Scan_Date": scan_date,
        "Image_ID": image_id,
        "Modality": modality,
        "Folder_Path": str(folder_path)
    }
    
    if scan_info not in data:
        data.append(scan_info)

scans_df = pd.DataFrame(data)
print(f"Found {len(scans_df)} unique scans. Now mapping to labels...\n")


# ---------------------------------------------------------
# PART 2: MATCH WITH DXSUM FOR LABELS
# ---------------------------------------------------------
dxsum_path = r"D:\ADNIzip1\DXSUM_16Apr2026.csv"  

dxsum_df = pd.read_csv(dxsum_path)

# Convert dates, turning unreadable garbage into NaT (Not a Time)
dxsum_df['EXAMDATE'] = pd.to_datetime(dxsum_df['EXAMDATE'], errors='coerce')
scans_df['Scan_Date'] = pd.to_datetime(scans_df['Scan_Date'], errors='coerce')

def get_closest_label(row):
    patient_id = row['PTID']
    scan_date = row['Scan_Date']
    
    patient_history = dxsum_df[dxsum_df['PTID'] == patient_id].copy()
    
    if patient_history.empty or pd.isna(scan_date):
        return pd.Series({'Diagnosis_Label': None, 'Days_Difference': None, 'Visit_Code': None})
    
    patient_history['Date_Diff'] = abs(patient_history['EXAMDATE'] - scan_date)
    
    # The ultimate safety net: Try to find the minimum, catch the error if it's empty
    try:
        closest_idx = patient_history['Date_Diff'].idxmin()
        closest_visit = patient_history.loc[closest_idx]
        
        return pd.Series({
            'Diagnosis_Label': closest_visit['DIAGNOSIS'], 
            'Days_Difference': closest_visit['Date_Diff'].days,
            'Visit_Code': closest_visit['VISCODE']
        })
    except Exception as e:
        return pd.Series({'Diagnosis_Label': None, 'Days_Difference': None, 'Visit_Code': None})

scans_df[['Diagnosis_Label', 'Days_Difference', 'Visit_Code']] = scans_df.apply(get_closest_label, axis=1)

# ---------------------------------------------------------
# PART 3: SAVE THE RESULTS
# ---------------------------------------------------------
output_file = r"D:\MinorCode\final_training_labels.csv"
scans_df.to_csv(output_file, index=False)

print("SUCCESS! File saved to:", output_file)
print("\nPreview of mapped data:")
print(scans_df[['PTID', 'Scan_Date', 'Diagnosis_Label', 'Days_Difference']].head())