# Program Kategorisasi Part Number berdasarkan Demand Pattern - VERSI FINAL
# Kriteria di kategoriparts.py mengikuti standar akademik yang umum dipakai 
# untuk demand intermittent: klasifikasi Syntetos–Boylan (SB)
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def _find_col(df, candidates):
    """Cari kolom secara fleksibel (case/space insensitive)."""
    norm_map = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower().replace(" ", "_")
        if key in norm_map:
            return norm_map[key]
    return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename kolom input ke standar: PART_NO, MONTH, PART_NAME, ORIGINAL_SHIPPING_QTY."""
    part_no_col = _find_col(df, [
        'PART_NO', 'PART NO', 'Part No', 'PartNo', 'PARTNUMBER', 'PART_NUMBER'
    ])
    month_col = _find_col(df, [
        'MONTH', 'Order Month', 'ORDER_MONTH', 'Order_Month', 'BULAN', 'TANGGAL'
    ])
    name_col = _find_col(df, [
        'PART_NAME', 'Part Name', 'Nama Part', 'ITEM_NAME'
    ])
    qty_col = _find_col(df, [
        'ORIGINAL_SHIPPING_QTY', 'Order Qty', 'ORDER_QTY', 'QTY', 'JUMLAH'
    ])
    rename_map = {}
    if part_no_col: rename_map[part_no_col] = 'PART_NO'
    if month_col: rename_map[month_col] = 'MONTH'
    if name_col: rename_map[name_col] = 'PART_NAME'
    if qty_col: rename_map[qty_col] = 'ORIGINAL_SHIPPING_QTY'
    df2 = df.rename(columns=rename_map).copy()
    # Validasi minimal kolom wajib
    required = ['PART_NO', 'MONTH', 'ORIGINAL_SHIPPING_QTY']
    missing = [c for c in required if c not in df2.columns]
    if missing:
        raise ValueError(f"❌ Kolom yang diperlukan tidak ditemukan setelah normalisasi: {missing}. Kolom tersedia: {list(df.columns)}")
    return df2

def parse_month_column(series):
    """
    Parse kolom MONTH yang formatnya YYYYMM (202401, 202402, dll) atau YYYY-MM ke datetime
    """
    try:
        # Jika sudah datetime, return
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        
        # Convert ke string dulu
        series_str = series.astype(str)
        
        # Debug info
        print(f"🔍 Parsing MONTH column:")
        print(f"   Sample values: {series_str.head(5).tolist()}")
        print(f"   Data type: {series.dtype}")
        
        # Handle format YYYYMM (202401, 202402, dll) - termasuk float seperti 202501.0
        if len(series_str) > 0 and series_str.iloc[0] != 'nan':
            # Cek apakah format YYYYMM atau YYYYMM.0
            first_val = str(series_str.iloc[0])
            if len(first_val) == 6 or (len(first_val) == 8 and first_val.endswith('.0')):
                print(f"   Detected format: YYYYMM (including float format)")
                # Handle float format seperti 202501.0
                if '.' in first_val:
                    series_clean = series_str.str.replace('.0', '', regex=False)
                else:
                    series_clean = series_str
                parsed_dates = pd.to_datetime(series_clean, format='%Y%m', errors='coerce')
                print(f"   Parsed sample: {parsed_dates.head(3).tolist()}")
                return parsed_dates
        
        # Handle format YYYY-MM
        elif series_str.str.contains('-', na=False).any():
            print(f"   Detected format: YYYY-MM")
            parsed_dates = pd.to_datetime(series_str, format='%Y-%m', errors='coerce')
            print(f"   Parsed sample: {parsed_dates.head(3).tolist()}")
            return parsed_dates
        
        # Handle format YYYY/MM
        elif series_str.str.contains('/', na=False).any():
            print(f"   Detected format: YYYY/MM")
            parsed_dates = pd.to_datetime(series_str, format='%Y/%m', errors='coerce')
            print(f"   Parsed sample: {parsed_dates.head(3).tolist()}")
            return parsed_dates
        
        # Handle format YYYY-MM-DD
        elif series_str.str.contains('-', na=False).any() and series_str.str.len().iloc[0] > 7:
            print(f"   Detected format: YYYY-MM-DD")
            parsed_dates = pd.to_datetime(series_str, format='%Y-%m-%d', errors='coerce')
            print(f"   Parsed sample: {parsed_dates.head(3).tolist()}")
            return parsed_dates
        
        # Fallback: auto parsing
        else:
            print(f"   Detected format: Auto parsing")
            parsed_dates = pd.to_datetime(series_str, errors='coerce')
            print(f"   Parsed sample: {parsed_dates.head(3).tolist()}")
            return parsed_dates
            
    except Exception as e:
        print(f"❌ Error parsing month column: {e}")
        print(f"   Sample data: {series.head(5).tolist()}")
        # Return original series as fallback
        return series

def calculate_adi_cv2_excel_style(series):
    """
    Menghitung ADI dan CV² sesuai dengan rumus Excel manual:
    
    ADI = COLUMNS(range) / COUNTIF(range,">0")
    CV² = (STDEV.P(range) / AVERAGE(range))^2
    
    Sesuai dengan perhitungan Excel manual yang benar
    """
    # Hapus nilai NaN
    series_clean = series.dropna()
    
    if len(series_clean) == 0:
        return np.nan, np.nan
    
    # Hitung ADI sesuai rumus Excel: COLUMNS(range) / COUNTIF(range,">0")
    total_periods = len(series_clean)  # COLUMNS(range)
    demand_periods = (series_clean > 0).sum()  # COUNTIF(range,">0")
    
    if demand_periods == 0:
        return np.nan, np.nan
    
    adi = total_periods / demand_periods
    
    # Hitung CV² sesuai rumus Excel: (STDEV.P(range) / AVERAGE(range))^2
    if len(series_clean) < 2:  # Minimal 2 data untuk hitung std
        cv2 = np.nan
    else:
        # Gunakan std populasi (ddof=0) seperti STDEV.P di Excel
        mean_demand = series_clean.mean()  # AVERAGE(range)
        std_demand = series_clean.std(ddof=0)  # STDEV.P(range) - ddof=0 untuk populasi
        
        if mean_demand == 0:
            cv2 = np.nan
        else:
            cv = std_demand / mean_demand
            cv2 = cv ** 2
    
    return adi, cv2

def validate_adi_cv2_calculation_excel_style(series, part_no):
    """
    Validasi perhitungan ADI dan CV² sesuai rumus Excel untuk debugging
    """
    print(f"\n🔍 VALIDASI PERHITUNGAN EXCEL STYLE untuk Part {part_no}:")
    print(f"   Data length: {len(series)}")
    print(f"   Sample data: {series.head(10).tolist()}")
    print(f"   Non-zero periods: {(series > 0).sum()}")
    print(f"   Zero periods: {(series == 0).sum()}")
    print(f"   Mean (all data): {series.mean():.3f}")
    print(f"   Std (all data): {series.std():.3f}")
    print(f"   Std (populasi, ddof=0): {series.std(ddof=0):.3f}")
    
    # Hitung ADI sesuai rumus Excel: COLUMNS(range) / COUNTIF(range,">0")
    total_periods = len(series)  # COLUMNS(range)
    demand_periods = (series > 0).sum()  # COUNTIF(range,">0")
    adi_excel = total_periods / demand_periods if demand_periods > 0 else np.nan
    print(f"   ADI Excel style: {total_periods}/{demand_periods} = {adi_excel:.3f}")
    
    # Hitung CV² sesuai rumus Excel: (STDEV.P(range) / AVERAGE(range))^2
    if len(series) >= 2 and series.mean() > 0:
        mean_excel = series.mean()  # AVERAGE(range)
        std_excel = series.std(ddof=0)  # STDEV.P(range)
        cv_excel = std_excel / mean_excel
        cv2_excel = cv_excel ** 2
        print(f"   CV² Excel style: ({std_excel:.3f}/{mean_excel:.3f})² = {cv2_excel:.3f}")
    else:
        print(f"   CV² Excel style: Cannot calculate (insufficient data or mean=0)")

def classify_demand_pattern_excel_style(adi, cv2):
    """
    Klasifikasi sesuai dengan rumus Excel manual:
    =IF(AND(M5<1.32,N5<0.49),"Smooth",
       IF(AND(M5>=1.32,N5<0.49),"Intermittent",
       IF(AND(M5<1.32,N5>=0.49),"Erratic",
       IF(AND(M5>=1.32,N5>=0.49),"Lumpy",""))))
    """
    if pd.isna(adi) or pd.isna(cv2):
        return 'Unknown'
    
    # Kriteria klasifikasi sesuai rumus Excel
    if adi < 1.32 and cv2 < 0.49:
        return 'Smooth'
    elif adi >= 1.32 and cv2 < 0.49:
        return 'Intermittent'
    elif adi < 1.32 and cv2 >= 0.49:
        return 'Erratic'
    elif adi >= 1.32 and cv2 >= 0.49:
        return 'Lumpy'
    else:
        return 'Unknown'

def categorize_parts_by_demand_corrected(df):
    """
    Kategorisasi yang BENAR untuk setiap part number
    """
    print("🔄 Memulai proses kategorisasi demand pattern...")
    
    # Pastikan kolom yang diperlukan ada
    required_cols = ['PART_NO', 'MONTH', 'ORIGINAL_SHIPPING_QTY']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Kolom yang diperlukan tidak ditemukan: {missing_cols}")
    
    # Parse kolom MONTH dengan benar ke format YYYY-MM
    print("📅 Parsing kolom MONTH ke format YYYY-MM...")
    df['MONTH'] = parse_month_column(df['MONTH'])
    
    # Cek hasil parsing
    print(f"   Sample MONTH setelah parsing: {df['MONTH'].head(3).tolist()}")
    
    # Sort berdasarkan PART_NO dan MONTH (SIMPAN SELURUH DATA untuk output)
    df_sorted_full = df.sort_values(['PART_NO', 'MONTH']).reset_index(drop=True)
    
    # LOGIKA PEMILIHAN DATA YANG BENAR
    print("🗓️ Menentukan periode data untuk perhitungan kategori...")
    
    # Parse MONTH column dengan format yang fleksibel
    print(f"   Format MONTH asli: {df_sorted_full['MONTH'].dtype}")
    print(f"   Sample MONTH values: {df_sorted_full['MONTH'].head(3).tolist()}")
    
    if df_sorted_full['MONTH'].dtype == 'object':
        # Jika format YYYY-MM, langsung gunakan
        if df_sorted_full['MONTH'].str.contains('-').any():
            print("   Detected format: YYYY-MM")
            month_dt_all = pd.to_datetime(df_sorted_full['MONTH'], format='%Y-%m', errors='coerce')
        else:
            # Jika format lain, gunakan auto parsing
            print("   Detected format: Other, using auto parsing")
            month_dt_all = pd.to_datetime(df_sorted_full['MONTH'], errors='coerce')
    else:
        print("   Detected format: Non-object, using auto parsing")
        month_dt_all = pd.to_datetime(df_sorted_full['MONTH'], errors='coerce')
    
    # Cek hasil parsing
    print(f"   Sample parsed dates: {month_dt_all.head(3).tolist()}")
    print(f"   Parsed date range: {month_dt_all.min()} hingga {month_dt_all.max()}")
    
    # LOGIKA PEMILIHAN DATA YANG BENAR:
    # Jika dataset dimulai SEBELUM 2022 → gunakan data >= 2022
    # Jika dataset dimulai 2022 atau SETELAHNYA → gunakan SEMUA data
    min_date = month_dt_all.min()
    print(f"   Tanggal minimum dataset: {min_date}")
    
    if min_date < pd.Timestamp('2022-01-01'):
        print("   Dataset dimulai SEBELUM 2022 → menggunakan data >= 2022-01-01")
        mask_for_category = month_dt_all >= pd.Timestamp('2022-01-01')
    else:
        print("   Dataset dimulai 2022 atau SETELAHNYA → menggunakan SEMUA data")
        mask_for_category = pd.Series([True] * len(df_sorted_full), index=df_sorted_full.index)
    
    df_sorted_for_category = df_sorted_full.loc[mask_for_category].copy()
    print(f"   Baris untuk perhitungan kategori: {len(df_sorted_for_category)} dari total {len(df_sorted_full)}")
    print(f"   Data dimulai dari: {df_sorted_for_category['MONTH'].min()} hingga {df_sorted_for_category['MONTH'].max()}")

    # Tentukan bulan terakhir untuk perhitungan
    max_month = month_dt_all.max()
    if pd.isna(max_month):
        print(f"   ❌ Error: Tidak ada data tanggal yang valid!")
        raise ValueError("Tidak ada data tanggal yang valid untuk kategorisasi")
    print(f"   Bulan terakhir dataset: {max_month.strftime('%Y-%m')}")
    
    # PERBAIKAN: Ambil data UNIQUE per PART_NO untuk kategorisasi
    print("🔍 Mengambil data unique per PART_NO...")
    
    # Buat DataFrame untuk kategorisasi (1 row per PART_NO)
    unique_parts_data = []
    
    # Ambil daftar part dari SELURUH data (bukan yang sudah difilter),
    # supaya semua part tetap muncul di output, tetapi kategorinya dihitung dari >=2022.
    unique_parts = df_sorted_full['PART_NO'].unique()
    total_parts = len(unique_parts)
    
    print(f"📊 Total UNIQUE part number yang akan dikategorisasi: {total_parts}")
    
    for idx, part_no in enumerate(unique_parts, 1):
        if idx % 50 == 0:
            print(f"   Progress: {idx}/{total_parts} parts ({idx/total_parts*100:.1f}%)")
        
        # Data untuk perhitungan kategori: gunakan SEMUA periode data
        part_data_cat = df_sorted_for_category[df_sorted_for_category['PART_NO'] == part_no].sort_values('MONTH')
        part_data_cat['MONTH_DT'] = pd.to_datetime(part_data_cat['MONTH'], format='%Y-%m')
        part_data_monthly = pd.Series(dtype=float)
        if not part_data_cat.empty:
            # PERBAIKAN: Gunakan SEMUA periode data dengan resampling yang benar
            # Resample untuk mengisi periode yang hilang dengan 0
            part_data_monthly = (
                part_data_cat
                .set_index('MONTH_DT')
                .resample('MS')
                .sum(numeric_only=True)
                ['ORIGINAL_SHIPPING_QTY']
                .fillna(0)
            )
            
            # PERBAIKAN: Pastikan menggunakan rentang data yang benar
            # Jika data dimulai 2025-01, gunakan dari 2025-01 sampai akhir data
            min_date_part = part_data_cat['MONTH_DT'].min()
            max_date_part = part_data_cat['MONTH_DT'].max()
            
            # Buat rentang lengkap dari min_date_part sampai max_date_part
            full_range = pd.date_range(start=min_date_part, end=max_date_part, freq='MS')
            part_data_monthly = part_data_monthly.reindex(full_range, fill_value=0)

        
        # Debug: tampilkan sample data untuk part ini
        if idx <= 3:  # Tampilkan untuk 3 part pertama
            print(f"\n🔍 Debug Part {part_no}:")
            print(f"   Data asli (sebelum resample): {len(part_data_cat)} records")
            print(f"   Sample data asli: {part_data_cat[['MONTH', 'ORIGINAL_SHIPPING_QTY']].head(5).values.tolist()}")
            if pd.isna(min_date_part) or pd.isna(max_date_part):
                print(f"   ❌ Error: Invalid date range for part {part_no}")
                continue
            print(f"   Rentang data: {min_date_part.strftime('%Y-%m')} sampai {max_date_part.strftime('%Y-%m')}")
            print(f"   Data length (setelah resample): {len(part_data_monthly)}")
            print(f"   Sample demand (setelah resample): {part_data_monthly.head(5).tolist()}")
            print(f"   Non-zero demand: {(part_data_monthly > 0).sum()}")
            print(f"   Zero demand: {(part_data_monthly == 0).sum()}")
            
            # Validasi perhitungan untuk debugging
            validate_adi_cv2_calculation_excel_style(part_data_monthly, part_no)
        
        # Hitung ADI dan CV² sesuai rumus Excel
        adi, cv2 = calculate_adi_cv2_excel_style(part_data_monthly)
        
        # ALTERNATIF: Perhitungan langsung dari data asli untuk perbandingan
        if idx <= 3:  # Debug untuk 3 part pertama
            print(f"\n🔍 ALTERNATIF PERHITUNGAN (data asli tanpa resample):")
            if not part_data_cat.empty:
                # Hitung langsung dari data asli (tanpa resample) sesuai rumus Excel
                original_demand = part_data_cat['ORIGINAL_SHIPPING_QTY'].values
                original_adi = len(original_demand) / (original_demand > 0).sum() if (original_demand > 0).sum() > 0 else np.nan
                original_cv2 = (original_demand.std(ddof=0) / original_demand.mean()) ** 2 if original_demand.mean() > 0 else np.nan
                print(f"   Original ADI: {original_adi:.3f}")
                print(f"   Original CV²: {original_cv2:.3f}")
                print(f"   Original data length: {len(original_demand)}")
                print(f"   Original non-zero: {(original_demand > 0).sum()}")
                print(f"   Original zero: {(original_demand == 0).sum()}")
                
                # Klasifikasi berdasarkan data asli sesuai rumus Excel
                if not pd.isna(original_adi) and not pd.isna(original_cv2):
                    original_category = classify_demand_pattern_excel_style(original_adi, original_cv2)
                    print(f"   Original Category: {original_category}")
        
        # Klasifikasikan demand pattern sesuai rumus Excel
        category = classify_demand_pattern_excel_style(adi, cv2)
        
        # Simpan data untuk part ini (1 row per PART_NO)
        # Ambil nama part dari SELURUH data agar konsisten dengan output
        part_data_full = df_sorted_full[df_sorted_full['PART_NO'] == part_no]
        unique_parts_data.append({
            'PART_NO': part_no,
            'PART_NAME': part_data_full['PART_NAME'].iloc[0] if 'PART_NAME' in part_data_full.columns and not part_data_full.empty else 'N/A',
            'ADI': adi,
            'CV2': cv2,
            'PartCategory': category,
            'Total_Months': int(len(part_data_monthly)),
            'Months_With_Demand': int((part_data_monthly > 0).sum()),
            'Total_Demand': float(part_data_monthly.sum()) if len(part_data_monthly) else 0.0,
            'Avg_Demand': float(part_data_monthly.mean()) if len(part_data_monthly) else 0.0
        })
        
        # Debug: tampilkan hasil perhitungan dengan detail
        if idx <= 5:  # Tampilkan untuk 5 part pertama
            print(f"   ADI: {adi:.3f} (total_periods={len(part_data_monthly)}, demand_periods={(part_data_monthly > 0).sum()})")
            print(f"   CV²: {cv2:.3f} (mean={part_data_monthly.mean():.2f}, std={part_data_monthly.std():.2f})")
            print(f"   Category: {category}")
            
            # Analisis mengapa masuk kategori tertentu
            if adi >= 1.32:
                print(f"   → ADI >= 1.32: Irregular time (demand jarang muncul)")
            else:
                print(f"   → ADI < 1.32: Regular time (demand sering muncul)")
            
            if cv2 >= 0.49:
                print(f"   → CV² >= 0.49: Irregular quantity (variabilitas tinggi)")
            else:
                print(f"   → CV² < 0.49: Regular quantity (variabilitas rendah)")
    
    # Buat DataFrame untuk kategorisasi
    df_categorized_unique = pd.DataFrame(unique_parts_data)
    
    # Update DataFrame ASLI (SELURUH riwayat) dengan kategori dari hasil perhitungan >=2022
    df_sorted_full['PartCategory'] = df_sorted_full['PART_NO'].map(
        df_categorized_unique.set_index('PART_NO')['PartCategory']
    )
    
    # Filter data untuk output: gunakan logika yang sama dengan perhitungan kategori
    print("📅 Filtering output data...")
    df_output = df_sorted_full[mask_for_category].copy()
    print(f"   Output data: {len(df_output)} baris (dari {len(df_sorted_full)} total)")
    
    # Hitung summary per kategori (berdasarkan UNIQUE PART_NO)
    category_counts = df_categorized_unique['PartCategory'].value_counts()
    
    print(f"\n✅ Kategorisasi selesai!")
    print(f"📊 Distribusi kategori (berdasarkan UNIQUE PART_NO):")
    for category, count in category_counts.items():
        percentage = (count / total_parts) * 100
        print(f"   - {category}: {count} unique parts ({percentage:.1f}%)")
    
    # Debug: tampilkan sample ADI dan CV²
    print(f"\n�� Sample ADI dan CV²:")
    sample_parts = df_categorized_unique.head(5)
    for _, row in sample_parts.iterrows():
        print(f"   Part {row['PART_NO']}: ADI={row['ADI']:.3f}, CV²={row['CV2']:.3f}, Category={row['PartCategory']}")
    
    # Analisis distribusi ADI dan CV² untuk memahami mengapa semua Intermittent/Lumpy
    print(f"\n📊 ANALISIS DISTRIBUSI ADI dan CV²:")
    print(f"   ADI Statistics:")
    print(f"     - Min: {df_categorized_unique['ADI'].min():.3f}")
    print(f"     - Max: {df_categorized_unique['ADI'].max():.3f}")
    print(f"     - Mean: {df_categorized_unique['ADI'].mean():.3f}")
    print(f"     - Median: {df_categorized_unique['ADI'].median():.3f}")
    print(f"     - Parts dengan ADI < 1.32: {(df_categorized_unique['ADI'] < 1.32).sum()}")
    print(f"     - Parts dengan ADI >= 1.32: {(df_categorized_unique['ADI'] >= 1.32).sum()}")
    
    print(f"   CV² Statistics:")
    print(f"     - Min: {df_categorized_unique['CV2'].min():.3f}")
    print(f"     - Max: {df_categorized_unique['CV2'].max():.3f}")
    print(f"     - Mean: {df_categorized_unique['CV2'].mean():.3f}")
    print(f"     - Median: {df_categorized_unique['CV2'].median():.3f}")
    print(f"     - Parts dengan CV² < 0.49: {(df_categorized_unique['CV2'] < 0.49).sum()}")
    print(f"     - Parts dengan CV² >= 0.49: {(df_categorized_unique['CV2'] >= 0.49).sum()}")
    
    # Analisis kombinasi ADI dan CV²
    print(f"\n🔍 ANALISIS KOMBINASI ADI dan CV²:")
    smooth_count = ((df_categorized_unique['ADI'] < 1.32) & (df_categorized_unique['CV2'] < 0.49)).sum()
    erratic_count = ((df_categorized_unique['ADI'] < 1.32) & (df_categorized_unique['CV2'] >= 0.49)).sum()
    intermittent_count = ((df_categorized_unique['ADI'] >= 1.32) & (df_categorized_unique['CV2'] < 0.49)).sum()
    lumpy_count = ((df_categorized_unique['ADI'] >= 1.32) & (df_categorized_unique['CV2'] >= 0.49)).sum()
    
    print(f"   Smooth (ADI<1.32, CV²<0.49): {smooth_count}")
    print(f"   Erratic (ADI<1.32, CV²>=0.49): {erratic_count}")
    print(f"   Intermittent (ADI>=1.32, CV²<0.49): {intermittent_count}")
    print(f"   Lumpy (ADI>=1.32, CV²>=0.49): {lumpy_count}")
    
    # Jika semua Intermittent/Lumpy, analisis penyebab
    if smooth_count == 0 and erratic_count == 0:
        print(f"\n⚠️  PERINGATAN: Tidak ada part yang masuk kategori Smooth atau Erratic!")
        print(f"   Kemungkinan penyebab:")
        print(f"   1. ADI terlalu tinggi (banyak periode dengan demand 0)")
        print(f"   2. CV² terlalu tinggi (variabilitas demand sangat tinggi)")
        print(f"   3. Ambang batas standar (1.32, 0.49) terlalu ketat untuk data ini")
        print(f"   4. Data preprocessing mungkin menghilangkan informasi penting")
    
    # KEMBALIKAN: data 2022+ + kolom PartCategory, serta tabel unik kategori
    return df_output, df_categorized_unique, category_counts

def create_excel_output_corrected(df_categorized, df_unique_categorized, filename='partcategory.xlsx'):
    """
    Membuat file Excel dengan multiple sheets sesuai permintaan
    """
    print(f"\n Membuat file Excel: {filename}")
    
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Dataset 2022+ + kolom PartCategory (TANPA ADI dan CV2)
            print("   📋 Sheet 1: Dataset 2022+ + PartCategory")
            # Hapus kolom ADI dan CV2 jika ada
            df_sheet1 = df_categorized.drop(columns=['ADI', 'CV2'], errors='ignore')
            df_sheet1.to_excel(writer, sheet_name='Dataset_2022_Plus_Category', index=False)
            
            # Sheet 2: Part number dengan kategori Smooth (data 2022+)
            print("   📋 Sheet 2: Part number Smooth")
            smooth_parts = df_categorized[df_categorized['PartCategory'] == 'Smooth']
            if not smooth_parts.empty:
                # Hapus kolom ADI dan CV2, hanya tambahkan PartCategory
                smooth_sheet = smooth_parts.drop(columns=['ADI', 'CV2'], errors='ignore')
                smooth_sheet.to_excel(writer, sheet_name='Smooth_Parts', index=False)
                print(f"      ✅ Smooth: {len(smooth_parts['PART_NO'].unique())} unique parts")
            else:
                pd.DataFrame({'Message': ['Tidak ada part number dengan kategori Smooth']}).to_excel(
                    writer, sheet_name='Smooth_Parts', index=False)
                print(f"      ⚠️  Smooth: 0 parts")
            
            # Sheet 3: Part number dengan kategori Erratic (data 2022+)
            print("   📋 Sheet 3: Part number Erratic")
            erratic_parts = df_categorized[df_categorized['PartCategory'] == 'Erratic']
            if not erratic_parts.empty:
                # Hapus kolom ADI dan CV2, hanya tambahkan PartCategory
                erratic_sheet = erratic_parts.drop(columns=['ADI', 'CV2'], errors='ignore')
                erratic_sheet.to_excel(writer, sheet_name='Erratic_Parts', index=False)
                print(f"      ✅ Erratic: {len(erratic_parts['PART_NO'].unique())} unique parts")
            else:
                pd.DataFrame({'Message': ['Tidak ada part number dengan kategori Erratic']}).to_excel(
                    writer, sheet_name='Erratic_Parts', index=False)
                print(f"      ⚠️  Erratic: 0 parts")
            
            # Sheet 4: Part number dengan kategori Intermittent (data 2022+)
            print("   📋 Sheet 4: Part number Intermittent")
            intermittent_parts = df_categorized[df_categorized['PartCategory'] == 'Intermittent']
            if not intermittent_parts.empty:
                # Hapus kolom ADI dan CV2, hanya tambahkan PartCategory
                intermittent_sheet = intermittent_parts.drop(columns=['ADI', 'CV2'], errors='ignore')
                intermittent_sheet.to_excel(writer, sheet_name='Intermittent_Parts', index=False)
                print(f"      ✅ Intermittent: {len(intermittent_parts['PART_NO'].unique())} unique parts")
            else:
                pd.DataFrame({'Message': ['Tidak ada part number dengan kategori Intermittent']}).to_excel(
                    writer, sheet_name='Intermittent_Parts', index=False)
                print(f"      ⚠️  Intermittent: 0 parts")
            
            # Sheet 5: Part number dengan kategori Lumpy (data 2022+)
            print("   📋 Sheet 5: Part number Lumpy")
            lumpy_parts = df_categorized[df_categorized['PartCategory'] == 'Lumpy']
            if not lumpy_parts.empty:
                # Hapus kolom ADI dan CV2, hanya tambahkan PartCategory
                lumpy_sheet = lumpy_parts.drop(columns=['ADI', 'CV2'], errors='ignore')
                lumpy_sheet.to_excel(writer, sheet_name='Lumpy_Parts', index=False)
                print(f"      ✅ Lumpy: {len(lumpy_parts['PART_NO'].unique())} unique parts")
            else:
                pd.DataFrame({'Message': ['Tidak ada part number dengan kategori Lumpy']}).to_excel(
                    writer, sheet_name='Lumpy_Parts', index=False)
                print(f"      ⚠️  Lumpy: 0 parts")
            
            # Sheet 6: Part number dengan kategori Unknown (data 2022+)
            print("   📋 Sheet 6: Part number Unknown")
            unknown_parts = df_categorized[df_categorized['PartCategory'] == 'Unknown']
            if not unknown_parts.empty:
                # Hapus kolom ADI dan CV2, hanya tambahkan PartCategory
                unknown_sheet = unknown_parts.drop(columns=['ADI', 'CV2'], errors='ignore')
                unknown_sheet.to_excel(writer, sheet_name='Unknown_Parts', index=False)
                print(f"      ⚠️  Unknown: {len(unknown_parts['PART_NO'].unique())} unique parts")
            else:
                pd.DataFrame({'Message': ['Tidak ada part number dengan kategori Unknown']}).to_excel(
                    writer, sheet_name='Unknown_Parts', index=False)
                print(f"      ✅ Unknown: 0 parts")
            
            # Sheet 7: Summary dan Analisis
            print("   📋 Sheet 7: Summary dan Analisis")
            summary_data = []
            
            # Summary per kategori
            category_counts = df_unique_categorized['PartCategory'].value_counts()
            for category, count in category_counts.items():
                percentage = (count / len(df_unique_categorized)) * 100
                summary_data.append({
                    'Kategori': category,
                    'Jumlah_Unique_Part': count,
                    'Persentase': f"{percentage:.1f}%"
                })
            
            # Tambahkan total
            total_parts = len(df_unique_categorized)
            summary_data.append({
                'Kategori': 'TOTAL',
                'Jumlah_Unique_Part': total_parts,
                'Persentase': '100.0%'
            })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary_Analysis', index=False)
            
            # Sheet 8: Statistik ADI dan CV² per kategori
            print("   Sheet 8: Statistik ADI dan CV² per kategori")
            stats_data = []
            
            for category in ['Smooth', 'Erratic', 'Intermittent', 'Lumpy']:
                category_data = df_unique_categorized[df_unique_categorized['PartCategory'] == category]
                if not category_data.empty:
                    stats_data.append({
                        'Kategori': category,
                        'Jumlah_Unique_Part': len(category_data),
                        'ADI_Rata_Rata': category_data['ADI'].mean(),
                        'ADI_Std': category_data['ADI'].std(),
                        'CV2_Rata_Rata': category_data['CV2'].mean(),
                        'CV2_Std': category_data['CV2'].std()
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Statistics_ADI_CV2', index=False)
            else:
                pd.DataFrame({'Message': ['Tidak ada data untuk statistik']}).to_excel(
                    writer, sheet_name='Statistics_ADI_CV2', index=False)
            
            # Sheet 9: Analisis Part Unknown (penyebab dan rekomendasi)
            print("   📋 Sheet 9: Analisis Part Unknown")
            unknown_analysis = []
            
            if not unknown_parts.empty:
                # Ambil data unique untuk analisis
                unknown_unique = df_unique_categorized[df_unique_categorized['PartCategory'] == 'Unknown']
                
                for _, row in unknown_unique.iterrows():
                    part_no = row['PART_NO']
                    adi = row['ADI']
                    cv2 = row['CV2']
                    
                    # Analisis penyebab "Unknown"
                    if pd.isna(adi):
                        reason = "ADI = NaN (tidak ada demand sama sekali)"
                        recommendation = "Part tidak aktif, pertimbangkan untuk di-deactivate"
                    elif pd.isna(cv2):
                        reason = "CV² = NaN (insufficient data untuk perhitungan)"
                        recommendation = "Data tidak cukup, perlu data historis lebih lengkap"
                    else:
                        reason = "Tidak masuk kriteria klasifikasi yang ada"
                        recommendation = "Perlu review manual berdasarkan business knowledge"
                    
                    unknown_analysis.append({
                        'PART_NO': part_no,
                        'PART_NAME': row['PART_NAME'],
                        'ADI': adi,
                        'CV2': cv2,
                        'Total_Months': row['Total_Months'],
                        'Months_With_Demand': row['Months_With_Demand'],
                        'Total_Demand': row['Total_Demand'],
                        'Avg_Demand': row['Avg_Demand'],
                        'Penyebab_Unknown': reason,
                        'Rekomendasi': recommendation
                    })
                
                unknown_analysis_df = pd.DataFrame(unknown_analysis)
                unknown_analysis_df.to_excel(writer, sheet_name='Unknown_Analysis', index=False)
                print(f"      ✅ Unknown Analysis: {len(unknown_analysis)} parts dianalisis")
            else:
                pd.DataFrame({'Message': ['Tidak ada part number dengan kategori Unknown untuk dianalisis']}).to_excel(
                    writer, sheet_name='Unknown_Analysis', index=False)
                print(f"      ✅ Unknown Analysis: 0 parts")
        
        print(f"✅ File Excel berhasil dibuat: {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error saat membuat file Excel: {str(e)}")
        return False

def main_corrected():
    """
    Fungsi utama yang sudah diperbaiki
    """
    print("🚀 PROGRAM KATEGORISASI PART NUMBER BERDASARKAN DEMAND PATTERN - VERSI FINAL")
    print("=" * 80)
    
    try:
        # 1. Load dataset
        file_path = 'TCO-dataset.xlsx'
        print(f"📁 Loading dataset dari: {file_path}")
        
        df = pd.read_excel(file_path)
        # Normalisasi nama kolom ke standar
        df = normalize_columns(df)
        print(f"✅ Dataset berhasil dimuat")
        print(f"�� Shape dataset: {df.shape}")
        print(f"📋 Kolom (setelah normalisasi): {list(df.columns)}")
        
        # 2. Validasi kolom yang diperlukan
        required_cols = ['PART_NO', 'MONTH', 'ORIGINAL_SHIPPING_QTY']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Kolom yang diperlukan tidak ditemukan: {missing_cols}")
        
        # 3. Tampilkan sample data sebelum parsing
        print(f"\n🔍 Sample data SEBELUM parsing MONTH (5 baris pertama):")
        print(df[required_cols].head())
        
        # 4. Cek format MONTH asli
        print(f"\n🔍 Cek format MONTH asli:")
        print(f"   - Sample MONTH: {df['MONTH'].head(3).tolist()}")
        print(f"   - Tipe data MONTH: {df['MONTH'].dtype}")
        print(f"   - Sample unique MONTH: {df['MONTH'].unique()[:5]}")
        
        # 5. Kategorisasi demand pattern
        print(f"\n🔄 Memulai proses kategorisasi...")
        df_categorized, df_unique_categorized, category_summary = categorize_parts_by_demand_corrected(df)
        
        # 6. Tampilkan hasil kategorisasi
        print(f"\n✅ HASIL KATEGORISASI (berdasarkan UNIQUE PART_NO):")
        print("-" * 50)
        for category, count in category_summary.items():
            percentage = (count / len(df_unique_categorized)) * 100
            print(f"   {category}: {count} unique parts ({percentage:.1f}%)")
        
        # 7. Buat file Excel output
        print(f"\n�� Membuat file Excel output...")
        success = create_excel_output_corrected(df_categorized, df_unique_categorized, 'TCO-categorize.xlsx')
        
        if success:
            print(f"\n🎉 PROGRAM BERHASIL SELESAI!")
            print(f"📁 File output: TCO-categorize.xlsx")
            print(f"📊 Total UNIQUE part number yang dikategorisasi: {len(df_unique_categorized)}")
            
            # Tampilkan rekomendasi berdasarkan kategori
            print(f"\n�� REKOMENDASI BERDASARKAN KATEGORI:")
            print("-" * 50)
            
            if 'Smooth' in category_summary:
                print(f"✅ Smooth ({category_summary['Smooth']} parts):")
                print(f"   - Karakteristik: ADI < 1.32, CV² < 0.49")
                print(f"   - Demand: Regular time, regular quantity")
                print(f"   - Forecast: Easy to forecast, low error level")
                print(f"   - Rekomendasi: Gunakan metode tradisional (ETS, ARIMA, MA)")
            
            if 'Intermittent' in category_summary:
                print(f"⚠️  Intermittent ({category_summary['Intermittent']} parts):")
                print(f"   - Karakteristik: ADI >= 1.32, CV² < 0.49")
                print(f"   - Demand: Irregular time, regular quantity")
                print(f"   - Forecast: High forecast error margin")
                print(f"   - Rekomendasi: Gunakan metode khusus (Croston, SBA, TSB)")
            
            if 'Erratic' in category_summary:
                print(f"⚠️  Erratic ({category_summary['Erratic']} parts):")
                print(f"   - Karakteristik: ADI < 1.32, CV² >= 0.49")
                print(f"   - Demand: Regular time, irregular quantity")
                print(f"   - Forecast: Shaky forecast accuracy")
                print(f"   - Rekomendasi: Gunakan ML methods (LSTM, RF, XGB)")
            
            if 'Lumpy' in category_summary:
                print(f"❌ Lumpy ({category_summary['Lumpy']} parts):")
                print(f"   - Karakteristik: ADI >= 1.32, CV² >= 0.49")
                print(f"   - Demand: Irregular time, irregular quantity")
                print(f"   - Forecast: Impossible to produce reliable forecast")
                print(f"   - Rekomendasi: Unforecastable, pertimbangkan safety stock")
        else:
            print(f"\n❌ Program gagal membuat file Excel output")
            
    except FileNotFoundError:
        print(f"❌ File tidak ditemukan: {file_path}")
        print("�� Pastikan file 'TCO-dataset.xlsx' ada di direktori kerja saat ini")
    except Exception as e:
        print(f"❌ Error dalam program: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

def filter_data_history(df, max_months=None, min_months=12):
    """
    Filter data history sesuai requirement:
    - Maksimal 48 bulan (4 tahun) data terakhir
    - Minimal 12 bulan data terakhir
    - Jika data > 48 bulan, ambil 48 bulan terakhir
    - Jika data < 12 bulan, gunakan semua data yang ada
    """
    import os
    
    # Get max_months from environment variable or use default
    if max_months is None:
        # Batas default 24 bulan (2 tahun) agar lebih cepat
        max_months = int(os.environ.get('MAX_DATA_MONTHS', '24'))
    
    print(f"📅 Filtering data history...")
    print(f"   Max months: {max_months} (4 years)")
    print(f"   Min months: {min_months}")
    
    # Parse MONTH column
    df['MONTH_DT'] = pd.to_datetime(df['MONTH'], format='%Y-%m', errors='coerce')
    
    # Cek data yang valid
    valid_data = df.dropna(subset=['MONTH_DT'])
    if len(valid_data) == 0:
        print("⚠️  No valid date data found")
        return df
    
    # Urutkan berdasarkan tanggal
    valid_data = valid_data.sort_values('MONTH_DT')
    
    # Ambil unique months
    unique_months = valid_data['MONTH_DT'].dt.to_period('M').unique()
    total_months = len(unique_months)
    
    print(f"   Total months in data: {total_months}")
    
    if total_months < min_months:
        print(f"   Data has {total_months} months (< {min_months}), using all available data")
        return valid_data
    elif total_months <= max_months:
        print(f"   Data has {total_months} months (≤ {max_months}), using all data")
        return valid_data
    else:
        # Ambil max_months terakhir (48 bulan = 4 tahun)
        print(f"   Data has {total_months} months (> {max_months} = 4 years), taking last {max_months} months")
        cutoff_date = unique_months[-max_months].start_time
        filtered_data = valid_data[valid_data['MONTH_DT'] >= cutoff_date]
        if pd.isna(cutoff_date):
            print(f"   ❌ Error: Invalid cutoff date")
            return valid_data
        print(f"   Filtered to {len(filtered_data)} records from {cutoff_date.strftime('%Y-%m')}")
        return filtered_data

def process_categorization(df):
    """
    Fungsi utama untuk kategorisasi yang digunakan di Railway
    Sesuai dengan alur operasional yang sudah ada
    """
    import os
    import tempfile
    from datetime import datetime
    
    print("🚀 Starting parts categorization...")
    
    try:
        # 1. Validasi input
        if df is None or df.empty:
            raise ValueError("❌ Input dataframe is empty")
        
        print(f"📊 Input data shape: {df.shape}")
        print(f"📋 Available columns: {list(df.columns)}")
        
        # 2. Normalisasi kolom
        df_normalized = normalize_columns(df)
        print(f"✅ Columns normalized")
        
        # 3. Parse MONTH column dengan pengecekan yang lebih robust
        print("📅 Parsing kolom MONTH ke format datetime...")
        original_month_col = df_normalized['MONTH'].copy()
        
        # Coba parsing MONTH
        try:
            df_normalized['MONTH'] = parse_month_column(df_normalized['MONTH'])
            
            # Cek apakah parsing berhasil
            if df_normalized['MONTH'].isna().all():
                print("❌ Error: Semua data MONTH gagal di-parse!")
                print(f"   Sample original MONTH values: {original_month_col.head(10).tolist()}")
                print(f"   Sample MONTH data types: {original_month_col.head(5).apply(type).tolist()}")
                raise ValueError("Gagal parsing kolom MONTH - semua data menjadi NaT")
            
            # Cek berapa banyak yang berhasil di-parse
            parsed_count = df_normalized['MONTH'].notna().sum()
            total_count = len(df_normalized)
            print(f"✅ MONTH parsing berhasil: {parsed_count}/{total_count} records")
            
            if parsed_count < total_count:
                print(f"⚠️ Warning: {total_count - parsed_count} records gagal di-parse")
                # Filter hanya data yang berhasil di-parse
                df_normalized = df_normalized[df_normalized['MONTH'].notna()].copy()
                print(f"📊 Data setelah filtering: {len(df_normalized)} records")
            
        except Exception as e:
            print(f"❌ Error parsing MONTH column: {e}")
            print(f"   Sample original MONTH values: {original_month_col.head(10).tolist()}")
            raise ValueError(f"Gagal parsing kolom MONTH: {e}")
        
        # 4. Filter data history (maksimal 24 bulan terakhir, minimal 12 bulan)
        df_filtered = filter_data_history(df_normalized)
        print(f"📅 Data filtered to {len(df_filtered)} records")
        
        # 5. Kategorisasi demand pattern dengan optimasi
        print("🔄 Starting optimized demand pattern categorization...")
        
        # Memory cleanup sebelum kategorisasi
        import gc
        gc.collect()
        
        df_categorized, df_unique_categorized, category_summary = categorize_parts_by_demand_corrected(df_filtered)
        
        # Memory cleanup setelah kategorisasi
        gc.collect()
        print("🧹 Memory cleanup completed after categorization")
        
        # 5. Tampilkan hasil
        print(f"\n✅ CATEGORIZATION COMPLETED:")
        print(f"📊 Total unique parts: {len(df_unique_categorized)}")
        for category, count in category_summary.items():
            percentage = (count / len(df_unique_categorized)) * 100
            print(f"   {category}: {count} parts ({percentage:.1f}%)")
        
        # 6. Buat file Excel output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"categorization_result_{timestamp}.xlsx"
        
        # Gunakan temporary directory untuk Railway
        temp_dir = tempfile.gettempdir()
        excel_path = os.path.join(temp_dir, excel_filename)
        
        success = create_excel_output_corrected(df_categorized, df_unique_categorized, excel_path)
        
        if success:
            print(f"✅ Excel file created: {excel_path}")
            return df_categorized, excel_path
        else:
            print("⚠️  Excel creation failed, returning data without file")
            return df_categorized, None
            
    except Exception as e:
        print(f"❌ Error in categorization: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        # Return original data with Unknown category
        df_result = df.copy()
        df_result['PART_CATEGORY'] = 'Unknown'
        return df_result, None

# Jalankan program yang sudah diperbaiki
if __name__ == "__main__":
    main_corrected()