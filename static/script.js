console.log('script.js loaded');

// Inisialisasi elemen untuk download forecast workflow
const processBtn = document.getElementById('process-btn');
const categorizeBtn = document.getElementById('categorize-btn');
const fileInput = document.getElementById('download-file-input');
const statusDiv = document.getElementById('download-status');
const progressBarContainer = document.getElementById('download-progress-bar-container');
const progressBar = document.getElementById('download-progress-bar');
const progressLabel = document.getElementById('download-progress-label');

// Fungsi validasi file Excel
function validateExcelFile(file) {
  const fileName = file.name.toLowerCase();
  const validExtensions = ['.xlsx'];
  const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext));
  
  if (!hasValidExtension) {
    return {
      isValid: false,
      message: 'Tipe File Kamu Bukan Excel. Pastikan File Tipe Excel Saja yang Kamu Unggah!'
    };
  }
  
  const validMimeTypes = [
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel'
  ];
  
  if (file.type && !validMimeTypes.includes(file.type)) {
    return {
      isValid: false,
      message: 'Tipe File Kamu Bukan Excel. Pastikan File Tipe Excel Saja yang Kamu Unggah!'
    };
  }
  
  return {
    isValid: true,
    message: ''
  };
}

// Update state tombol berdasarkan file yang dipilih
function updateProcessBtnState() {
  if (processBtn && categorizeBtn && fileInput) {
    const hasFile = fileInput.files.length > 0;
    const isValidFile = hasFile && validateExcelFile(fileInput.files[0]).isValid;
    
    // Update tombol forecast (yang memerlukan file)
    processBtn.disabled = !hasFile || !isValidFile;
    
    // Tombol kategorisasi tidak memerlukan file (bisa upload file terpisah)
    categorizeBtn.disabled = false;
    
    statusDiv.className = '';
    
    if (hasFile && !isValidFile) {
      const validation = validateExcelFile(fileInput.files[0]);
      statusDiv.textContent = validation.message;
      statusDiv.classList.add('error');
    } else if (hasFile && isValidFile) {
      statusDiv.textContent = 'File Excel valid, siap untuk diproses.';
      statusDiv.classList.add('success');
    } else {
      statusDiv.textContent = '';
    }
  }
}

// Event listener untuk file input
if (fileInput) {
  fileInput.addEventListener('change', updateProcessBtnState);
}

// Fungsi untuk kategorisasi terpisah
async function processCategorization() {
  // Buat file input terpisah untuk kategorisasi
  const categorizeFileInput = document.createElement('input');
  categorizeFileInput.type = 'file';
  categorizeFileInput.accept = '.xlsx';
  categorizeFileInput.style.display = 'none';
  document.body.appendChild(categorizeFileInput);
  
  // Trigger file picker
  categorizeFileInput.click();
  
  categorizeFileInput.addEventListener('change', async function() {
    if (!categorizeFileInput.files.length) {
      statusDiv.textContent = 'Tidak ada file yang dipilih.';
      statusDiv.classList.add('error');
      document.body.removeChild(categorizeFileInput);
      return;
    }

    const validation = validateExcelFile(categorizeFileInput.files[0]);
    if (!validation.isValid) {
      statusDiv.textContent = validation.message;
      statusDiv.classList.add('error');
      document.body.removeChild(categorizeFileInput);
      return;
    }

    const formData = new FormData();
    formData.append('file', categorizeFileInput.files[0]);

  try {
    statusDiv.textContent = '📊 Memproses kategorisasi...';
    statusDiv.classList.add('info');
    progressBarContainer.style.display = 'block';
    progressBar.style.width = '20%';
    progressLabel.textContent = 'Uploading file...';

    const response = await fetch('/run-categorization', {
      method: 'POST',
      body: formData
    });

    progressBar.style.width = '60%';
    progressLabel.textContent = 'Processing categorization...';

    if (response.ok) {
      const result = await response.json();
      progressBar.style.width = '100%';
      progressLabel.textContent = 'Complete!';
      
      if (result.status === 'success') {
        statusDiv.textContent = 'Kategorisasi Pola Demand Selesai';
        statusDiv.classList.add('success');
        
        // Download file hasil kategorisasi otomatis
        if (result.download_url) {
          window.open(result.download_url, '_blank');
        }
      } else {
        statusDiv.textContent = `❌ Error: ${result.message}`;
        statusDiv.classList.add('error');
      }
    } else {
      const errorData = await response.json();
      statusDiv.textContent = `❌ Error: ${errorData.detail || 'Server error'}`;
      statusDiv.classList.add('error');
    }
  } catch (error) {
    statusDiv.textContent = `❌ Error: ${error.message}`;
    statusDiv.classList.add('error');
  } finally {
    setTimeout(() => {
      progressBarContainer.style.display = 'none';
      progressBar.style.width = '0%';
      progressLabel.textContent = 'Processing...';
      document.body.removeChild(categorizeFileInput);
    }, 2000);
  }
  });
}

// Event listener untuk tombol kategorisasi
if (categorizeBtn) {
  categorizeBtn.addEventListener('click', processCategorization);
}

// ====== DASHBOARD INTERAKTIF DINAMIS ======
let originalData = [];
let backtestData = [];
let realtimeData = [];
let totalStats = {}; // Global variable untuk total aggregation statistics

// Chart.js instance
let realtimeBarChart = null;
let realtimeLineChart = null;
let backtestLineChart = null;
let backtestBarChart = null;

// Data dummy yang lebih lengkap untuk testing dengan data yang cukup untuk charts
const dummyOriginal = [
  { PART_NO: 'PART001', MONTH: '2024-01', ORIGINAL_SHIPPING_QTY: 100 },
  { PART_NO: 'PART001', MONTH: '2024-02', ORIGINAL_SHIPPING_QTY: 120 },
  { PART_NO: 'PART001', MONTH: '2024-03', ORIGINAL_SHIPPING_QTY: 110 },
  { PART_NO: 'PART001', MONTH: '2024-04', ORIGINAL_SHIPPING_QTY: 130 },
  { PART_NO: 'PART001', MONTH: '2024-05', ORIGINAL_SHIPPING_QTY: 140 },
  { PART_NO: 'PART001', MONTH: '2024-06', ORIGINAL_SHIPPING_QTY: 125 },
  { PART_NO: 'PART001', MONTH: '2024-07', ORIGINAL_SHIPPING_QTY: 135 },
  { PART_NO: 'PART001', MONTH: '2024-08', ORIGINAL_SHIPPING_QTY: 145 },
  { PART_NO: 'PART001', MONTH: '2024-09', ORIGINAL_SHIPPING_QTY: 155 },
  { PART_NO: 'PART001', MONTH: '2024-10', ORIGINAL_SHIPPING_QTY: 165 },
  { PART_NO: 'PART001', MONTH: '2024-11', ORIGINAL_SHIPPING_QTY: 175 },
  { PART_NO: 'PART001', MONTH: '2024-12', ORIGINAL_SHIPPING_QTY: 185 },
  { PART_NO: 'PART002', MONTH: '2024-01', ORIGINAL_SHIPPING_QTY: 200 },
  { PART_NO: 'PART002', MONTH: '2024-02', ORIGINAL_SHIPPING_QTY: 220 },
  { PART_NO: 'PART002', MONTH: '2024-03', ORIGINAL_SHIPPING_QTY: 210 },
  { PART_NO: 'PART002', MONTH: '2024-04', ORIGINAL_SHIPPING_QTY: 230 },
  { PART_NO: 'PART002', MONTH: '2024-05', ORIGINAL_SHIPPING_QTY: 240 },
  { PART_NO: 'PART002', MONTH: '2024-06', ORIGINAL_SHIPPING_QTY: 225 },
  { PART_NO: 'PART002', MONTH: '2024-07', ORIGINAL_SHIPPING_QTY: 235 },
  { PART_NO: 'PART002', MONTH: '2024-08', ORIGINAL_SHIPPING_QTY: 245 },
  { PART_NO: 'PART002', MONTH: '2024-09', ORIGINAL_SHIPPING_QTY: 255 },
  { PART_NO: 'PART002', MONTH: '2024-10', ORIGINAL_SHIPPING_QTY: 265 },
  { PART_NO: 'PART002', MONTH: '2024-11', ORIGINAL_SHIPPING_QTY: 275 },
  { PART_NO: 'PART002', MONTH: '2024-12', ORIGINAL_SHIPPING_QTY: 285 },
  { PART_NO: 'PART003', MONTH: '2024-01', ORIGINAL_SHIPPING_QTY: 150 },
  { PART_NO: 'PART003', MONTH: '2024-02', ORIGINAL_SHIPPING_QTY: 170 },
  { PART_NO: 'PART003', MONTH: '2024-03', ORIGINAL_SHIPPING_QTY: 160 },
  { PART_NO: 'PART003', MONTH: '2024-04', ORIGINAL_SHIPPING_QTY: 180 },
  { PART_NO: 'PART003', MONTH: '2024-05', ORIGINAL_SHIPPING_QTY: 190 },
  { PART_NO: 'PART003', MONTH: '2024-06', ORIGINAL_SHIPPING_QTY: 175 },
  { PART_NO: 'PART003', MONTH: '2024-07', ORIGINAL_SHIPPING_QTY: 185 },
  { PART_NO: 'PART003', MONTH: '2024-08', ORIGINAL_SHIPPING_QTY: 195 },
  { PART_NO: 'PART003', MONTH: '2024-09', ORIGINAL_SHIPPING_QTY: 205 },
  { PART_NO: 'PART003', MONTH: '2024-10', ORIGINAL_SHIPPING_QTY: 215 },
  { PART_NO: 'PART003', MONTH: '2024-11', ORIGINAL_SHIPPING_QTY: 225 },
  { PART_NO: 'PART003', MONTH: '2024-12', ORIGINAL_SHIPPING_QTY: 235 }
];

const dummyRealtime = [
  { PART_NO: 'PART001', MONTH: '2025-01', FORECAST_OPTIMIST: 513557, FORECAST: 446571, FORECAST_PESSIMIST: 357257, INVENTORY_CONTROL_CLASS: 'A1' },
  { PART_NO: 'PART001', MONTH: '2025-02', FORECAST_OPTIMIST: 526700, FORECAST: 458000, FORECAST_PESSIMIST: 366400, INVENTORY_CONTROL_CLASS: 'A1' },
  { PART_NO: 'PART001', MONTH: '2025-03', FORECAST_OPTIMIST: 540500, FORECAST: 470000, FORECAST_PESSIMIST: 376000, INVENTORY_CONTROL_CLASS: 'A1' },
  { PART_NO: 'PART001', MONTH: '2025-04', FORECAST_OPTIMIST: 557750, FORECAST: 485000, FORECAST_PESSIMIST: 388000, INVENTORY_CONTROL_CLASS: 'A1' },
  { PART_NO: 'PART002', MONTH: '2025-01', FORECAST_OPTIMIST: 594000, FORECAST: 550000, FORECAST_PESSIMIST: 506000, INVENTORY_CONTROL_CLASS: 'B2' },
  { PART_NO: 'PART002', MONTH: '2025-02', FORECAST_OPTIMIST: 615600, FORECAST: 570000, FORECAST_PESSIMIST: 524400, INVENTORY_CONTROL_CLASS: 'B2' },
  { PART_NO: 'PART002', MONTH: '2025-03', FORECAST_OPTIMIST: 637200, FORECAST: 590000, FORECAST_PESSIMIST: 542800, INVENTORY_CONTROL_CLASS: 'B2' },
  { PART_NO: 'PART002', MONTH: '2025-04', FORECAST_OPTIMIST: 658800, FORECAST: 610000, FORECAST_PESSIMIST: 561200, INVENTORY_CONTROL_CLASS: 'B2' },
  { PART_NO: 'PART003', MONTH: '2025-01', FORECAST_OPTIMIST: 368000, FORECAST: 320000, FORECAST_PESSIMIST: 256000, INVENTORY_CONTROL_CLASS: '' },
  { PART_NO: 'PART003', MONTH: '2025-02', FORECAST_OPTIMIST: 379500, FORECAST: 330000, FORECAST_PESSIMIST: 264000, INVENTORY_CONTROL_CLASS: '' },
  { PART_NO: 'PART003', MONTH: '2025-03', FORECAST_OPTIMIST: 391000, FORECAST: 340000, FORECAST_PESSIMIST: 272000, INVENTORY_CONTROL_CLASS: '' },
  { PART_NO: 'PART003', MONTH: '2025-04', FORECAST_OPTIMIST: 402500, FORECAST: 350000, FORECAST_PESSIMIST: 280000, INVENTORY_CONTROL_CLASS: '' }
];

const dummyBacktest = [
  { PART_NO: 'PART001', MONTH: '2024-09', FORECAST: 441515, ACTUAL: 155, HYBRID_ERROR: '18.54%', BEST_MODEL: 'WMA' },
  { PART_NO: 'PART001', MONTH: '2024-10', FORECAST: 450000, ACTUAL: 165, HYBRID_ERROR: '15.20%', BEST_MODEL: 'RF' },
  { PART_NO: 'PART001', MONTH: '2024-11', FORECAST: 460000, ACTUAL: 175, HYBRID_ERROR: '12.80%', BEST_MODEL: 'XGB' },
  { PART_NO: 'PART001', MONTH: '2024-12', FORECAST: 470000, ACTUAL: 185, HYBRID_ERROR: '10.50%', BEST_MODEL: 'WMA' },
  { PART_NO: 'PART002', MONTH: '2024-09', FORECAST: 550000, ACTUAL: 255, HYBRID_ERROR: '16.80%', BEST_MODEL: 'RF' },
  { PART_NO: 'PART002', MONTH: '2024-10', FORECAST: 560000, ACTUAL: 265, HYBRID_ERROR: '14.20%', BEST_MODEL: 'XGB' },
  { PART_NO: 'PART002', MONTH: '2024-11', FORECAST: 570000, ACTUAL: 275, HYBRID_ERROR: '11.50%', BEST_MODEL: 'WMA' },
  { PART_NO: 'PART002', MONTH: '2024-12', FORECAST: 580000, ACTUAL: 285, HYBRID_ERROR: '9.80%', BEST_MODEL: 'RF' }
];

// Helper: konversi ke format YYYY-MM
function toYearMonth(str) {
  if (!str) return '';
  if (typeof str === 'string' && str.length >= 7) return str.slice(0, 7);
  if (str instanceof Date) return str.getFullYear() + '-' + String(str.getMonth() + 1).padStart(2, '0');
  return str;
}

// Helper: format bulan untuk display
function formatMonthForDisplay(monthStr) {
  if (!monthStr) return '';
  const yearMonth = toYearMonth(monthStr);
  const [year, month] = yearMonth.split('-');
  return `${year}-${month}`;
}

// Update realtime cards berdasarkan filter
function updateRealtimeCards(filteredData, allCardsData) {
  // ICC Card - tampilkan ICC dari part pertama atau strip jika tidak ada
  const iccCard = document.getElementById('card-icc-value');
  const iccCardTitle = document.getElementById('card-icc');
  
  if (filteredData.length > 0) {
    // Ada filter part number - tampilkan ICC dari part pertama
    const firstPart = filteredData[0];
    let iccValue = '-';
    
    // Cek kolom ICC dengan berbagai kemungkinan nama
    if (firstPart.INVENTORY_CONTROL_CLASS && firstPart.INVENTORY_CONTROL_CLASS !== '' && firstPart.INVENTORY_CONTROL_CLASS !== '-') {
      iccValue = firstPart.INVENTORY_CONTROL_CLASS;
    } else if (firstPart.ICC && firstPart.ICC !== '' && firstPart.ICC !== '-') {
      iccValue = firstPart.ICC;
    } else if (firstPart.Inventory_Control_Class && firstPart.Inventory_Control_Class !== '' && firstPart.Inventory_Control_Class !== '-') {
      iccValue = firstPart.Inventory_Control_Class;
    }
    
    if (iccCard) iccCard.textContent = iccValue;
  } else {
    // Tidak ada filter - tampilkan strip
    if (iccCard) iccCard.textContent = '-';
  }
  
  // Forecast Cards - PERBAIKAN LOGIKA SESUAI SPESIFIKASI:
  // Cards forecast mengambil data dari sheet realtime hasil forecast (FORECAST_NEUTRAL)
  // Jika ada filter part number, gunakan data terfilter dari sheet realtime
  // Jika tidak ada filter part number, gunakan total_stats dari sheet realtime
  let forecastTotals = {};
  
  if (filteredData.length > 0) {
    // Ada filter part number - hitung dari data terfilter dari sheet realtime
    const uniqueMonths = [...new Set(filteredData.map(d => d.MONTH))].sort();
    uniqueMonths.forEach(month => {
      const monthData = filteredData.filter(d => d.MONTH === month);
      // Gunakan FORECAST_NEUTRAL dari sheet realtime hasil forecast
      forecastTotals[month] = monthData.reduce((sum, d) => sum + (Number(d.FORECAST_NEUTRAL) || Number(d.FORECAST) || 0), 0);
    });
    console.log('Using filtered realtime data for forecast cards:', forecastTotals);
  } else if (totalStats && totalStats.realtime_totals) {
    // Tidak ada filter part number - gunakan total aggregation dari backend (sheet realtime)
    forecastTotals = totalStats.realtime_totals;
    console.log('Using total_stats from realtime sheet for forecast cards:', forecastTotals);
  } else {
    // Fallback: hitung manual dari allCardsData (sheet realtime)
    const uniqueMonths = [...new Set(allCardsData.map(d => d.MONTH))].sort();
    uniqueMonths.forEach(month => {
      const monthData = allCardsData.filter(d => d.MONTH === month);
      // Gunakan FORECAST_NEUTRAL dari sheet realtime hasil forecast
      forecastTotals[month] = monthData.reduce((sum, d) => sum + (Number(d.FORECAST_NEUTRAL) || Number(d.FORECAST) || 0), 0);
    });
    console.log('Using manual calculation from realtime sheet for forecast cards:', forecastTotals);
  }
  
  // Ambil 2 bulan pertama dari forecast totals
  const sortedMonths = Object.keys(forecastTotals).sort();
  const firstMonth = sortedMonths[0];
  const secondMonth = sortedMonths[1];
  
  // Update card titles
  const month1Title = document.getElementById('card-forecast-month1-title');
  const month2Title = document.getElementById('card-forecast-month2-title');
  
  if (month1Title) month1Title.textContent = firstMonth ? `Forecast ${formatMonthForDisplay(firstMonth)}` : 'Forecast';
  if (month2Title) month2Title.textContent = secondMonth ? `Forecast ${formatMonthForDisplay(secondMonth)}` : 'Forecast';
  
  // Update card values dengan total aggregation
  const month1Card = document.getElementById('card-forecast-month1-value');
  const month2Card = document.getElementById('card-forecast-month2-value');
  
  const totalMonth1 = forecastTotals[firstMonth] || 0;
  const totalMonth2 = forecastTotals[secondMonth] || 0;
  
  if (month1Card) month1Card.textContent = totalMonth1.toLocaleString();
  if (month2Card) month2Card.textContent = totalMonth2.toLocaleString();
}

// Fungsi filter data berdasarkan partno dan bulan
function getFilteredData(data, partno, selectedMonths) {
  let filtered = data;
  
  // Filter berdasarkan part number (jika ada input)
  if (partno && partno.trim() !== '') {
    filtered = filtered.filter(d => d.PART_NO && d.PART_NO.toLowerCase().includes(partno.toLowerCase()));
  }
  // Jika partno kosong, tampilkan semua data (tidak perlu filter)
  
  // Filter berdasarkan bulan (jika ada bulan yang dipilih)
  if (selectedMonths && selectedMonths.length > 0) {
    filtered = filtered.filter(d => selectedMonths.includes(toYearMonth(d.MONTH)));
  }
  // Jika tidak ada bulan yang dipilih, tampilkan semua bulan
  
  return filtered;
}



// ===== CUSTOM MONTH PICKER (LOCAL) =====
let selectedRealtimeMonths = [];
let selectedBacktestMonths = [];

function setupMonthPickers() {
  console.log('Setting up custom month pickers...');
  
  // Ambil bulan yang tersedia dari data (semua part number)
  const realtimeMonths = getAvailableMonths(realtimeData);
  const backtestMonths = getAvailableMonths(backtestData);
  
  console.log('Available realtime months:', realtimeMonths);
  console.log('Available backtest months:', backtestMonths);

  // Pastikan elemen input ada
  const realtimeMonthInput = document.getElementById('realtime-month-picker');
  const backtestMonthInput = document.getElementById('backtest-month-picker');
  
  console.log('Realtime input found:', !!realtimeMonthInput);
  console.log('Backtest input found:', !!backtestMonthInput);
  
  // Buat calendar untuk realtime
  if (realtimeMonthInput) {
    createMonthCalendar(realtimeMonthInput, realtimeMonths, 'realtime');
  }
  
  // Buat calendar untuk backtest
  if (backtestMonthInput) {
    createMonthCalendar(backtestMonthInput, backtestMonths, 'backtest');
  }

  // Setup clear buttons
  const realtimeClearBtn = document.getElementById('realtime-month-clear');
  const backtestClearBtn = document.getElementById('backtest-month-clear');
  
  if (realtimeClearBtn) {
    realtimeClearBtn.onclick = function() {
      selectedRealtimeMonths = [];
      if (realtimeMonthInput) {
        realtimeMonthInput.value = '';
      }
      renderRealtimeDashboard();
    };
  }
  
  if (backtestClearBtn) {
    backtestClearBtn.onclick = function() {
      selectedBacktestMonths = [];
      if (backtestMonthInput) {
        backtestMonthInput.value = '';
      }
      renderBacktestDashboard();
    };
  }
}

function createMonthCalendar(inputElement, availableMonths, type) {
  // Hapus event listener yang ada
  inputElement.onclick = null;
  inputElement.onchange = null;
  
  // Style input
  inputElement.style.cssText = `
    width: 100%;
    padding: 8px 12px;
    border: 1px solid #3578e5;
    border-radius: 4px;
    background-color: #181828;
    color: #fff;
    cursor: pointer;
    font-size: 14px;
  `;
  inputElement.placeholder = 'Pilih bulan...';
  inputElement.readOnly = true;
  
  // Buat calendar container dengan posisi yang tepat
  const calendarContainer = document.createElement('div');
  calendarContainer.className = 'month-calendar-container';
  calendarContainer.style.cssText = `
    position: absolute;
    top: 100%;
    left: 0;
    background-color: #181828;
    border: 1px solid #3578e5;
    border-radius: 4px;
    padding: 16px;
    z-index: 9999;
    display: none;
    min-width: 280px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    margin-top: 4px;
  `;
  
  // Parse available months untuk mendapatkan tahun dan bulan
  const monthData = {};
  availableMonths.forEach(month => {
    const [year, monthNum] = month.split('-');
    if (!monthData[year]) monthData[year] = [];
    monthData[year].push(parseInt(monthNum));
  });
  
  // Buat calendar untuk setiap tahun
  Object.keys(monthData).sort().forEach(year => {
    const yearContainer = document.createElement('div');
    yearContainer.style.cssText = `
      margin-bottom: 16px;
    `;
    
    // Tahun header
    const yearHeader = document.createElement('div');
    yearHeader.textContent = year;
    yearHeader.style.cssText = `
      font-size: 16px;
      font-weight: bold;
      color: #fff;
      margin-bottom: 8px;
      text-align: center;
    `;
    yearContainer.appendChild(yearHeader);
    
    // Grid bulan
    const monthGrid = document.createElement('div');
    monthGrid.style.cssText = `
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 4px;
    `;
    
    const monthNames = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];
    
    // Buat semua bulan (1-12)
    for (let month = 1; month <= 12; month++) {
      const monthButton = document.createElement('button');
      monthButton.textContent = monthNames[month - 1];
      monthButton.style.cssText = `
        padding: 8px 4px;
        border: 1px solid #2a2a2a;
        background-color: #2a2a2a;
        color: #666;
        cursor: not-allowed;
        font-size: 12px;
        border-radius: 2px;
        transition: all 0.2s ease;
      `;
      
      // Cek apakah bulan ini tersedia
      const isAvailable = monthData[year].includes(month);
      
      if (isAvailable) {
        monthButton.style.cssText = `
          padding: 8px 4px;
          border: 1px solid #3578e5;
          background-color: #181828;
          color: #fff;
          cursor: pointer;
          font-size: 12px;
          border-radius: 2px;
          transition: all 0.2s ease;
        `;
        
        monthButton.onmouseover = function() {
          this.style.backgroundColor = '#3578e5';
          this.style.borderColor = '#4a9eff';
        };
        
        monthButton.onmouseout = function() {
          this.style.backgroundColor = '#181828';
          this.style.borderColor = '#3578e5';
        };
        
        monthButton.onclick = function() {
          const selectedMonth = `${year}-${String(month).padStart(2, '0')}`;
          inputElement.value = selectedMonth;
          
          if (type === 'realtime') {
            selectedRealtimeMonths = [selectedMonth];
            renderRealtimeDashboard();
          } else {
            selectedBacktestMonths = [selectedMonth];
            renderBacktestDashboard();
          }
          
          calendarContainer.style.display = 'none';
        };
      }
      
      monthGrid.appendChild(monthButton);
    }
    
    yearContainer.appendChild(monthGrid);
    calendarContainer.appendChild(yearContainer);
  });
  
  // Insert calendar sebagai child dari parent container
  const parentContainer = inputElement.closest('.month-range-group') || inputElement.parentNode;
  
  // Pastikan parent container memiliki position relative
  if (parentContainer.style.position !== 'relative') {
    parentContainer.style.position = 'relative';
  }
  
  // Pastikan input element juga memiliki position relative
  inputElement.style.position = 'relative';
  
  parentContainer.appendChild(calendarContainer);
  
  // Toggle calendar saat input diklik
  inputElement.onclick = function(e) {
    e.stopPropagation(); // Mencegah event bubbling
    const isVisible = calendarContainer.style.display === 'block';
    
    if (!isVisible) {
      // Pastikan kalender muncul tepat di bawah input
      const inputRect = inputElement.getBoundingClientRect();
      calendarContainer.style.top = '100%';
      calendarContainer.style.left = '0';
      calendarContainer.style.position = 'absolute';
      calendarContainer.style.zIndex = '9999';
    }
    
    calendarContainer.style.display = isVisible ? 'none' : 'block';
  };
  
  // Tutup calendar saat klik di luar
  document.addEventListener('click', function(e) {
    if (!calendarContainer.contains(e.target) && e.target !== inputElement) {
      calendarContainer.style.display = 'none';
    }
  });
  
  // Pastikan calendar tidak muncul di bawah elemen lain
  calendarContainer.addEventListener('click', function(e) {
    e.stopPropagation();
  });
}

// Helper: ambil bulan unik dari data (semua part number)
function getAvailableMonths(data) {
  const months = new Set();
  data.forEach(d => {
    if (d.MONTH) months.add(toYearMonth(d.MONTH));
  });
  return Array.from(months).sort();
}

// Render dashboard real-time
function renderRealtimeDashboard() {
  try {
    console.log('=== RENDER REALTIME DASHBOARD START ===');
    
    // Pastikan data dummy tersedia jika data kosong
    if (realtimeData.length === 0) {
      realtimeData = dummyRealtime;
      console.log('Using dummy realtime data, length:', realtimeData.length);
    }
    if (originalData.length === 0) {
      originalData = dummyOriginal;
      console.log('Using dummy original data, length:', originalData.length);
    }
    
    // Pastikan canvas elements ada
    const barCanvasCheck = document.getElementById('realtime-bar-chart');
    const lineCanvasCheck = document.getElementById('realtime-line-chart');
    
    console.log('Canvas elements check:', {
      barCanvas: !!barCanvasCheck,
      lineCanvas: !!lineCanvasCheck,
      barCanvasId: barCanvasCheck?.id,
      lineCanvasId: lineCanvasCheck?.id
    });
    
    if (!barCanvasCheck || !lineCanvasCheck) {
      console.error('Canvas elements not found:', {
        barCanvas: !!barCanvasCheck,
        lineCanvas: !!lineCanvasCheck
      });
      return;
    }
    
    // Pastikan Chart.js tersedia
    if (typeof Chart === 'undefined') {
      console.error('Chart.js not available');
      return;
    }
    
    console.log('Chart.js available:', typeof Chart);
    
    const partno = document.getElementById('partno-input').value.trim();
    
    // PERBAIKAN LOGIKA FILTERING SESUAI SPESIFIKASI:
    // 1. Filter part number: berlaku untuk SEMUA (cards, bar chart, line chart)
    // 2. Filter bulan: hanya berlaku untuk BAR CHART di realtime
    
    // Filter part number untuk semua komponen
    let filteredByPart = realtimeData;
    if (partno && partno.trim() !== '') {
      filteredByPart = realtimeData.filter(d => d.PART_NO && d.PART_NO.toLowerCase().includes(partno.toLowerCase()));
    }
    
    // Update cards berdasarkan filter part number (TIDAK ada filter bulan untuk cards)
    updateRealtimeCards(filteredByPart, realtimeData);
    
    // Filter bulan hanya untuk bar chart
    let filteredForBarChart = filteredByPart;
    if (selectedRealtimeMonths && selectedRealtimeMonths.length > 0) {
      filteredForBarChart = filteredByPart.filter(d => selectedRealtimeMonths.includes(toYearMonth(d.MONTH)));
    }
    
    // PERBAIKAN: Bar chart menggunakan total forecast per bulan dari sheet realtime (FORECAST_NEUTRAL)
    const monthTotals = {};
    filteredForBarChart.forEach(d => {
      const month = toYearMonth(d.MONTH);
      if (!monthTotals[month]) {
        monthTotals[month] = 0;
      }
      // Gunakan FORECAST_NEUTRAL dari sheet realtime hasil forecast
      monthTotals[month] += (d.FORECAST_NEUTRAL || d.FORECAST || 0);
    });
    
    const barLabels = Object.keys(monthTotals).sort((a, b) => new Date(a) - new Date(b));
    const neutral = barLabels.map(month => monthTotals[month]);
    
    console.log('Creating bar chart with total data:', {
      barLabels: barLabels,
      neutral: neutral,
      dataLength: barLabels.length,
      monthTotals: monthTotals
    });
    
    if (realtimeBarChart) {
      console.log('Destroying existing realtime bar chart');
      realtimeBarChart.destroy();
    }
    
    const barCanvas = document.getElementById('realtime-bar-chart');
    console.log('Bar canvas found:', !!barCanvas, 'Chart.js available:', typeof Chart !== 'undefined');
    
    if (barCanvas && typeof Chart !== 'undefined') {
      // Pastikan canvas memiliki ukuran yang benar
      barCanvas.style.width = '100%';
      barCanvas.style.height = '180px';
      barCanvas.style.display = 'block';
      barCanvas.style.visibility = 'visible';
      
      const ctxBar = barCanvas.getContext('2d');
      console.log('Canvas context created:', !!ctxBar);
      
      try {
        realtimeBarChart = new Chart(ctxBar, {
          type: 'bar',
          data: {
            labels: barLabels,
            datasets: [
              { 
                label: 'Forecast', 
                data: neutral, 
                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
              }
            ]
          },
          options: { 
            responsive: true, 
            maintainAspectRatio: false,
            plugins: { 
              legend: { 
                position: 'top',
                labels: {
                  color: '#fff',
                  font: {
                    weight: 'bold'
                  }
                }
              } 
            },
            scales: {
              x: {
                ticks: {
                  color: '#fff',
                  font: {
                    weight: 'bold'
                  }
                }
              },
              y: {
                beginAtZero: true,
                ticks: {
                  color: '#fff',
                  font: {
                    weight: 'bold'
                  }
                }
              }
            }
          }
        });
        console.log('Realtime bar chart created successfully');
      } catch (error) {
        console.error('Error creating bar chart:', error);
      }
    } else {
      console.error('Failed to create realtime bar chart:', {
        barCanvas: !!barCanvas,
        chartJsAvailable: typeof Chart !== 'undefined'
      });
    }
    
    // Line chart (12 bulan terakhir: 8 bulan history + 4 bulan forecast, filter bulan TIDAK berlaku)
    let history = originalData;
    // Filter history berdasarkan part number (jika ada input)
    if (partno && partno.trim() !== '') {
      history = history.filter(d => d.PART_NO && d.PART_NO.toLowerCase().includes(partno.toLowerCase()));
    }
    // Jika partno kosong, gunakan semua data history
    const allHistoryMonths = Array.from(new Set(history.map(d => d.MONTH))).sort((a, b) => new Date(a) - new Date(b));
    const last8History = allHistoryMonths.slice(-8);
    history = history.filter(d => last8History.includes(d.MONTH));
    
    // Ambil semua bulan forecast yang tersedia (tidak terpengaruh filter bulan)
    const allForecastMonths = Array.from(new Set(realtimeData.map(d => d.MONTH))).sort((a, b) => new Date(a) - new Date(b));
    console.log('Available forecast months:', allForecastMonths);
    
    // Gabungkan bulan history dan forecast, pastikan tidak ada duplikasi
    const allMonths = [...last8History, ...allForecastMonths];
    const uniqueMonths = Array.from(new Set(allMonths)).sort((a, b) => new Date(a) - new Date(b));
    const lineLabels = uniqueMonths.map(toYearMonth);
    
    // PERBAIKAN: Hitung total per bulan untuk history dan forecast
    // PERBAIKAN: History: dari dataset asli (originalData) - untuk periode NON-realtime (8 bulan awal)
    const historyTotals = {};
    history.forEach(d => {
      const month = toYearMonth(d.MONTH);
      if (!historyTotals[month]) {
        historyTotals[month] = 0;
      }
      // Gunakan ORIGINAL_SHIPPING_QTY atau ORDER_QTY dari dataset awal yang diunggah
      historyTotals[month] += (d.ORIGINAL_SHIPPING_QTY || d.ORDER_QTY || 0);
    });
    
    // PERBAIKAN: Forecast: dari file hasil forecast (realtimeData) - kolom FORECAST_NEUTRAL untuk periode realtime
    const forecastTotals = {};
    let forecastDataForChart = realtimeData; // Gunakan semua data realtime
    
    // Filter berdasarkan part number hanya jika ada input part number
    if (partno && partno.trim() !== '') {
      forecastDataForChart = realtimeData.filter(d => d.PART_NO && d.PART_NO.toLowerCase().includes(partno.toLowerCase()));
    }
    
    forecastDataForChart.forEach(d => {
      const month = toYearMonth(d.MONTH);
      if (!forecastTotals[month]) {
        forecastTotals[month] = 0;
      }
      // PERBAIKAN: Gunakan FORECAST_NEUTRAL dari sheet realtime hasil forecast
      forecastTotals[month] += (d.FORECAST_NEUTRAL || d.FORECAST || 0);
    });
    
    console.log('Line chart data sources:', {
      historyDataLength: history.length,
      forecastDataLength: forecastDataForChart.length,
      totalRealtimeDataLength: realtimeData.length,
      partnoFilter: partno,
      note: 'History: dataset awal (ORIGINAL_SHIPPING_QTY), Forecast: sheet realtime (FORECAST_NEUTRAL)'
    });
    
    // DEBUG: Log forecast totals calculation
    console.log('🔍 DEBUG: forecastTotals calculation:', forecastTotals);
    console.log('🔍 DEBUG: forecastDataForChart sample:', forecastDataForChart.slice(0, 3));
    
    // DEBUG: Check if FORECAST_NEUTRAL values are 0
    const forecastNeutralValues = forecastDataForChart.map(d => d.FORECAST_NEUTRAL || 0);
    console.log('🔍 DEBUG: FORECAST_NEUTRAL values in forecastDataForChart:', forecastNeutralValues.slice(0, 10));
    console.log('🔍 DEBUG: Non-zero FORECAST_NEUTRAL count:', forecastNeutralValues.filter(v => v > 0).length);
    console.log('🔍 DEBUG: forecastTotals after calculation:', forecastTotals);
    
    const lineHistory = lineLabels.map(m => historyTotals[m] || null);
    const lineNeutral = lineLabels.map(m => forecastTotals[m] || null);
    
    console.log('Creating line chart with data:', {
      lineLabels: lineLabels,
      lineHistory: lineHistory,
      lineNeutral: lineNeutral,
      dataLength: lineLabels.length,
      historyTotals: historyTotals,
      forecastTotals: forecastTotals
    });
    
    // DEBUG: Log line chart data
    console.log('🔍 DEBUG: lineNeutral values:', lineNeutral);
    console.log('🔍 DEBUG: lineHistory values:', lineHistory);
    
    // DEBUG: Check if lineNeutral has any non-zero values
    const nonZeroNeutral = lineNeutral.filter(v => v !== null && v > 0);
    console.log('🔍 DEBUG: Non-zero lineNeutral values:', nonZeroNeutral);
    console.log('🔍 DEBUG: lineNeutral length:', lineNeutral.length);
    
    if (realtimeLineChart) {
      console.log('Destroying existing realtime line chart');
      realtimeLineChart.destroy();
    }
    
    const lineCanvas = document.getElementById('realtime-line-chart');
    console.log('Line canvas found:', !!lineCanvas, 'Chart.js available:', typeof Chart !== 'undefined');
    
    if (lineCanvas && typeof Chart !== 'undefined') {
      // Pastikan canvas memiliki ukuran yang benar
      lineCanvas.style.width = '100%';
      lineCanvas.style.height = '180px';
      lineCanvas.style.display = 'block';
      lineCanvas.style.visibility = 'visible';
      
      const ctxLine = lineCanvas.getContext('2d');
      console.log('Line canvas context created:', !!ctxLine);
      
      try {
        realtimeLineChart = new Chart(ctxLine, {
          type: 'line',
          data: {
            labels: lineLabels,
            datasets: [
              { 
                label: 'History', 
                data: lineHistory, 
                borderColor: '#aaa', 
                backgroundColor: 'rgba(200,200,200,0.1)', 
                tension: 0.2,
                borderWidth: 2
              },
              { 
                label: 'Forecast', 
                data: lineNeutral, 
                borderColor: '#00bcd4', 
                backgroundColor: 'rgba(0,188,212,0.1)', 
                tension: 0.2,
                borderWidth: 3
              }
            ]
          },
          options: { 
            responsive: true, 
            maintainAspectRatio: false,
            plugins: { 
              legend: { 
                position: 'top',
                labels: {
                  color: '#fff',
                  font: {
                    weight: 'bold'
                  }
                }
              } 
            },
            scales: {
              x: {
                ticks: {
                  color: '#fff',
                  font: {
                    weight: 'bold'
                  }
                }
              },
              y: {
                beginAtZero: true,
                ticks: {
                  color: '#fff',
                  font: {
                    weight: 'bold'
                  }
                }
              }
            }
          }
        });
        console.log('Realtime line chart created successfully');
      } catch (error) {
        console.error('Error creating line chart:', error);
      }
    } else {
      console.error('Failed to create realtime line chart:', {
        lineCanvas: !!lineCanvas,
        chartJsAvailable: typeof Chart !== 'undefined'
      });
    }
  } catch (error) {
    console.error('Error in renderRealtimeDashboard:', error);
    // Fallback ke data dummy jika ada error
    originalData = dummyOriginal;
    realtimeData = dummyRealtime;
    // Coba render lagi dengan data dummy
    setTimeout(() => {
      renderRealtimeDashboard();
    }, 100);
  }
}

// Render dashboard backtest
function renderBacktestDashboard() {
  try {
    console.log('=== RENDER BACKTEST DASHBOARD START ===');
    
    // Pastikan data dummy tersedia jika data kosong
    if (backtestData.length === 0) {
      backtestData = dummyBacktest;
      console.log('Using dummy backtest data, length:', backtestData.length);
    }
    
    // Pastikan canvas elements ada
    const barCanvasCheck = document.getElementById('backtest-bar-chart');
    const lineCanvasCheck = document.getElementById('backtest-line-chart');
    
    console.log('Backtest canvas elements check:', {
      barCanvas: !!barCanvasCheck,
      lineCanvas: !!lineCanvasCheck,
      barCanvasId: barCanvasCheck?.id,
      lineCanvasId: lineCanvasCheck?.id
    });
    
    if (!barCanvasCheck || !lineCanvasCheck) {
      console.error('Backtest canvas elements not found:', {
        barCanvas: !!barCanvasCheck,
        lineCanvas: !!lineCanvasCheck
      });
      return;
    }
    
    // Pastikan Chart.js tersedia
    if (typeof Chart === 'undefined') {
      console.error('Chart.js not available for backtest');
      return;
    }
    
    console.log('Chart.js available for backtest:', typeof Chart);
    
    const partno = document.getElementById('partno-input').value.trim();
    
    // PERBAIKAN LOGIKA FILTERING SESUAI SPESIFIKASI:
    // 1. Filter part number: berlaku untuk SEMUA (cards, bar chart, line chart)
    // 2. Filter bulan: berlaku untuk CARDS dan BAR CHART, TIDAK untuk LINE CHART
    
    // Filter part number untuk semua komponen
    let filteredByPart = backtestData;
    if (partno && partno.trim() !== '') {
      filteredByPart = backtestData.filter(d => d.PART_NO && d.PART_NO.toLowerCase().includes(partno.toLowerCase()));
    }
    
    // Filter bulan untuk cards dan bar chart (TIDAK untuk line chart)
    let filteredForCardsAndBar = filteredByPart;
    if (selectedBacktestMonths && selectedBacktestMonths.length > 0) {
      filteredForCardsAndBar = filteredByPart.filter(d => selectedBacktestMonths.includes(toYearMonth(d.MONTH)));
    }
    
    // PERBAIKAN: Cards menggunakan data dari sheet backtest hasil forecast
    let cardsData = filteredForCardsAndBar;
    
    let sumForecast, avgError;
    
    if (filteredByPart.length > 0) {
      // Ada filter part number - hitung dari data terfilter dari sheet backtest
      sumForecast = filteredForCardsAndBar.reduce((sum, d) => sum + (Number(d.FORECAST) || 0), 0);
      avgError = filteredForCardsAndBar.length > 0 ? (filteredForCardsAndBar.reduce((a, b) => a + (parseFloat((b.ERROR||'0').replace('%',''))||0), 0) / filteredForCardsAndBar.length) : 0;
      console.log('Using filtered backtest data for cards:', { sumForecast, avgError });
    } else if (totalStats && totalStats.forecast_totals) {
      // Tidak ada filter part number - gunakan total aggregation dari backend (sheet backtest)
      sumForecast = Object.values(totalStats.forecast_totals).reduce((sum, val) => sum + val, 0);
      avgError = totalStats.average_error || 0;
      console.log('Using total_stats from backtest sheet for cards:', { sumForecast, avgError });
    } else {
      // Fallback: hitung manual dari data terfilter dari sheet backtest
      sumForecast = cardsData.reduce((sum, d) => sum + (Number(d.FORECAST) || 0), 0);
      avgError = cardsData.length > 0 ? (cardsData.reduce((a, b) => a + (parseFloat((b.ERROR||'0').replace('%',''))||0), 0) / cardsData.length) : 0;
      console.log('Using manual calculation from backtest sheet for cards:', { sumForecast, avgError });
    }
    
    // Update backtest card values
    const qtyCard = document.getElementById('card-backtest-qty-value');
    const errorCard = document.getElementById('card-backtest-error-value');
    
    if (qtyCard) qtyCard.textContent = sumForecast.toLocaleString();
    if (errorCard) errorCard.textContent = avgError.toFixed(2) + '%';
    
    // PERBAIKAN: Bar chart menggunakan jumlah part no untuk tiap best model dari sheet backtest
    const modelCounts = {};
    
    // Gunakan data yang terfilter untuk bar chart dari sheet backtest
    filteredForCardsAndBar.forEach(d => {
      // Hitung semua BEST_MODEL yang ada dari data terfilter dari sheet backtest
      if (d.BEST_MODEL) {
        const modelName = d.BEST_MODEL.toString().trim();
        if (modelName !== '') {
          modelCounts[modelName] = (modelCounts[modelName] || 0) + 1;
        }
      }
    });
    const modelLabels = Object.keys(modelCounts);
    const modelData = Object.values(modelCounts);
    
    // Debug: log data yang akan digunakan untuk chart
    console.log('Model counts from backtest sheet:', modelCounts);
    console.log('Model labels:', modelLabels);
    console.log('Model data:', modelData);
    console.log('Note: Data dari sheet backtest hasil forecast (BEST_MODEL)');
    
    if (backtestBarChart) backtestBarChart.destroy();
    const barCanvas = document.getElementById('backtest-bar-chart');
    console.log('Backtest bar canvas found:', !!barCanvas, 'Chart.js available:', typeof Chart !== 'undefined');
    if (barCanvas && typeof Chart !== 'undefined') {
      // Pastikan canvas memiliki ukuran yang benar
      barCanvas.style.width = '100%';
      barCanvas.style.height = '180px';
      barCanvas.style.display = 'block';
      const ctxBar = barCanvas.getContext('2d');
      console.log('Creating backtest bar chart with data:', modelLabels, modelData);
      backtestBarChart = new Chart(ctxBar, {
        type: 'bar',
        data: {
          labels: modelLabels,
          datasets: [
            { label: 'Best Model', data: modelData, backgroundColor: 'rgba(54, 162, 235, 0.6)' }
          ]
        },
        options: { 
          responsive: true, 
          maintainAspectRatio: false,
          plugins: { 
            legend: { 
              display: false,
              labels: {
                color: '#fff',
                font: {
                  weight: 'bold'
                }
              }
            } 
          },
          scales: {
            x: {
              ticks: {
                color: '#fff',
                font: {
                  weight: 'bold'
                }
              }
            },
            y: {
              beginAtZero: true,
              ticks: {
                color: '#fff',
                font: {
                  weight: 'bold'
                }
              }
            }
          }
        }
      });
      console.log('Backtest bar chart created successfully');
    } else {
      console.error('Failed to create backtest bar chart:', {
        barCanvas: !!barCanvas,
        chartJsAvailable: typeof Chart !== 'undefined'
      });
    }
    
    // PERBAIKAN: Line chart menggunakan total forecast dan actual per bulan dari sheet backtest
    const monthsLine = Array.from(new Set(filteredByPart.map(d => d.MONTH))).sort((a, b) => new Date(a) - new Date(b));
    
    // Hitung total forecast dan actual per bulan dari sheet backtest hasil forecast
    const forecastTotals = {};
    const actualTotals = {};
    
    filteredByPart.forEach(d => {
      const month = d.MONTH;
      if (!forecastTotals[month]) {
        forecastTotals[month] = 0;
      }
      if (!actualTotals[month]) {
        actualTotals[month] = 0;
      }
      // Gunakan FORECAST dan ACTUAL dari sheet backtest hasil forecast
      forecastTotals[month] += (Number(d.FORECAST) || 0);
      actualTotals[month] += (Number(d.ACTUAL) || 0);
    });
    
    const forecast = monthsLine.map(m => forecastTotals[m] || null);
    const actual = monthsLine.map(m => actualTotals[m] || null);
    
    if (backtestLineChart) backtestLineChart.destroy();
    const lineCanvas = document.getElementById('backtest-line-chart');
    console.log('Backtest line canvas found:', !!lineCanvas, 'Chart.js available:', typeof Chart !== 'undefined');
    if (lineCanvas && typeof Chart !== 'undefined') {
      // Pastikan canvas memiliki ukuran yang benar
      lineCanvas.style.width = '100%';
      lineCanvas.style.height = '180px';
      lineCanvas.style.display = 'block';
      const ctxLine = lineCanvas.getContext('2d');
      console.log('Creating backtest line chart with data:', {
      monthsLine: monthsLine,
      forecast: forecast,
      actual: actual,
      note: 'Data dari sheet backtest hasil forecast (FORECAST dan ACTUAL)'
    });
      backtestLineChart = new Chart(ctxLine, {
        type: 'line',
        data: {
          labels: monthsLine.map(m => m.replace(/T.*$/, '')),
          datasets: [
            { label: 'Forecast', data: forecast, borderColor: '#2196f3', backgroundColor: 'rgba(33,150,243,0.1)', tension: 0.2 },
            { label: 'Actual', data: actual, borderColor: '#aaa', backgroundColor: 'rgba(200,200,200,0.1)', tension: 0.2 }
          ]
        },
        options: { 
          responsive: true, 
          maintainAspectRatio: false,
          plugins: { 
            legend: { 
              position: 'top',
              labels: {
                color: '#fff',
                font: {
                  weight: 'bold'
                }
              }
            } 
          },
          scales: {
            x: {
              ticks: {
                color: '#fff',
                font: {
                  weight: 'bold'
                }
              }
            },
            y: {
              beginAtZero: true,
              ticks: {
                color: '#fff',
                font: {
                  weight: 'bold'
                }
              }
            }
          }
        }
      });
      console.log('Backtest line chart created successfully');
    } else {
      console.error('Failed to create backtest line chart:', {
        lineCanvas: !!lineCanvas,
        chartJsAvailable: typeof Chart !== 'undefined'
      });
    }
  } catch (error) {
    console.error('Error in renderBacktestDashboard:', error);
    // Fallback ke data dummy jika ada error
    backtestData = dummyBacktest;
    // Coba render lagi dengan data dummy
    setTimeout(() => {
      renderBacktestDashboard();
    }, 100);
  }
}

// Event handler input partno
const partnoInput = document.getElementById('partno-input');
if (partnoInput) {
  // Pastikan input tidak disabled dan bisa diketik
  partnoInput.disabled = false;
  partnoInput.readOnly = false;
  partnoInput.style.pointerEvents = 'auto';
  partnoInput.style.cursor = 'text';
  
  partnoInput.addEventListener('input', function() {
    // Render ulang dashboard setiap kali input berubah
    renderRealtimeDashboard();
    renderBacktestDashboard();
  });
  
  // Tambahkan event listener untuk keyup juga
  partnoInput.addEventListener('keyup', function() {
    // Render ulang dashboard setiap kali key dilepas
    renderRealtimeDashboard();
    renderBacktestDashboard();
  });
  
  // Tambahkan event listener untuk change juga
  partnoInput.addEventListener('change', function() {
    // Render ulang dashboard ketika nilai berubah
    renderRealtimeDashboard();
    renderBacktestDashboard();
  });
  
  // Tambahkan event listener untuk focus
  partnoInput.addEventListener('focus', function() {
    this.style.borderColor = '#3578e5';
  });
  
  // Tambahkan event listener untuk blur
  partnoInput.addEventListener('blur', function() {
    this.style.borderColor = '#bfc9d1';
  });
}

// Event listener untuk DOM loaded
window.addEventListener('DOMContentLoaded', () => {
  if (processBtn) processBtn.disabled = true;
  updateProcessBtnState();
  
  // Inisialisasi dengan dummy data
  originalData = dummyOriginal;
  backtestData = dummyBacktest;
  realtimeData = dummyRealtime;
  
  console.log('Dashboard initialized with dummy data:', {
    originalData: originalData.length,
    backtestData: backtestData.length,
    realtimeData: realtimeData.length
  });
  
  // Pastikan filter input tidak disabled
  const partnoInput = document.getElementById('partno-input');
  if (partnoInput) {
    partnoInput.disabled = false;
    partnoInput.readOnly = false;
    partnoInput.style.pointerEvents = 'auto';
    partnoInput.style.cursor = 'text';
  }
  
  // Pastikan month picker input tidak disabled
  const realtimeMonthInput = document.getElementById('realtime-month-picker');
  const backtestMonthInput = document.getElementById('backtest-month-picker');
  
  if (realtimeMonthInput) {
    realtimeMonthInput.disabled = false;
    realtimeMonthInput.readOnly = true; // Tetap readonly untuk flatpickr
    realtimeMonthInput.style.pointerEvents = 'auto';
    realtimeMonthInput.style.cursor = 'pointer';
  }
  
  if (backtestMonthInput) {
    backtestMonthInput.disabled = false;
    backtestMonthInput.readOnly = true; // Tetap readonly untuk flatpickr
    backtestMonthInput.style.pointerEvents = 'auto';
    backtestMonthInput.style.cursor = 'pointer';
  }
  
  // Setup month pickers
  setupMonthPickers();
  
  // Fungsi untuk memastikan Chart.js dimuat sebelum render
  function ensureChartJsLoaded() {
    console.log('Checking Chart.js availability...');
    if (typeof Chart !== 'undefined') {
      console.log('✅ Chart.js loaded successfully');
      // Render dashboard setelah Chart.js dimuat
      console.log('Rendering dashboards...');
  renderRealtimeDashboard();
  renderBacktestDashboard();
    } else {
      console.log('⏳ Chart.js not loaded yet, retrying...');
      setTimeout(ensureChartJsLoaded, 100);
    }
  }
  
  // Coba render dashboard setelah delay singkat untuk memastikan Chart.js dimuat
  setTimeout(() => {
    console.log('🔄 First render attempt (500ms)');
    ensureChartJsLoaded();
  }, 500);
  
  // Juga coba render dashboard setelah delay yang lebih lama untuk memastikan semua elemen ter-load
  setTimeout(() => {
    console.log('🔄 Second render attempt (1000ms)');
      renderRealtimeDashboard();
      renderBacktestDashboard();
  }, 1000);
  
  // Tambahan: coba render lagi setelah 2 detik untuk memastikan Chart.js benar-benar dimuat
  setTimeout(() => {
    console.log('🔄 Final render attempt (2000ms)');
    if (typeof Chart !== 'undefined') {
      renderRealtimeDashboard();
      renderBacktestDashboard();
    } else {
      console.error('❌ Chart.js still not available after 2 seconds');
    }
  }, 2000);
  
  // Tambahan: coba render lagi setelah 3 detik sebagai fallback terakhir
  setTimeout(() => {
    console.log('🔄 Last resort render attempt (3000ms)');
    renderRealtimeDashboard();
    renderBacktestDashboard();
  }, 3000);
});

// PERBAIKAN: Tambahkan connection status checking
async function checkServerConnection() {
  try {
    const response = await fetch('/api/health', { 
      method: 'GET',
      signal: AbortSignal.timeout(5000) // 5 second timeout
    });
    return response.ok;
  } catch (error) {
    console.warn('⚠️ Server connection check failed:', error.message);
    return false;
  }
}

// PERBAIKAN: Tambahkan long polling untuk status forecast
async function pollForecastStatus(sessionId, maxAttempts = 1440) { // 2 jam dengan check setiap 5 detik
  console.log('🔍 Starting long polling for forecast status...');
  console.log(`📊 Max attempts: ${maxAttempts} (2 hours total)`);
  
  let attempts = 0;
  const checkInterval = 5000; // Check setiap 5 detik
  
  return new Promise((resolve, reject) => {
    const checkStatus = async () => {
      try {
        attempts++;
        console.log(`🔍 Status check attempt ${attempts}/${maxAttempts} for session: ${sessionId}`);
        
        const response = await fetch(`/api/forecast-progress/${sessionId}`);
        
        if (response.ok) {
          const data = await response.json();
          console.log('📊 Status response:', data);
          
          if (data.progress === 'completed' || data.file_id) {
            console.log('✅ Forecast completed, file ready');
            resolve(data);
            return;
          } else {
            console.log('⏳ Forecast still processing, progress:', data.progress);
          }
        } else {
          console.log(`❌ Response not OK: ${response.status}`);
        }
        
        if (attempts >= maxAttempts) {
          console.log('⏰ Max attempts reached, stopping status check');
          reject(new Error('Forecast timeout setelah 2 jam monitoring'));
          return;
        }
        
        // Continue checking
        console.log(`⏳ Waiting ${checkInterval/1000} seconds before next check...`);
        setTimeout(checkStatus, checkInterval);
        
      } catch (error) {
        console.error('❌ Error checking status:', error);
        if (attempts >= maxAttempts) {
          reject(error);
          return;
        }
        
        // Continue checking even if there's an error
        console.log(`⏳ Error occurred, but continuing... Next check in ${checkInterval/1000} seconds`);
        setTimeout(checkStatus, checkInterval);
      }
    };
    
    // Start checking
    console.log('🚀 Starting status check loop...');
    checkStatus();
  });
}

// Event listener untuk tombol proses forecast
if (processBtn && fileInput && statusDiv && progressBarContainer && progressBar && progressLabel) {
  processBtn.addEventListener('click', async function() {
    statusDiv.textContent = '';
    statusDiv.className = '';
    
    // PERBAIKAN: Check server connection sebelum mulai
    console.log('🔍 Checking server connection...');
    const isServerConnected = await checkServerConnection();
    if (!isServerConnected) {
      statusDiv.textContent = '⚠️ Server tidak dapat diakses. Periksa koneksi internet atau coba refresh halaman.';
      statusDiv.classList.add('error');
      return;
    }
    console.log('✅ Server connection OK, proceeding with forecast...');
    
    if (!fileInput.files.length) {
      statusDiv.textContent = 'Pilih file Excel terlebih dahulu.';
      statusDiv.classList.add('error');
      return;
    }

    const validation = validateExcelFile(fileInput.files[0]);
    if (!validation.isValid) {
      statusDiv.textContent = validation.message;
      statusDiv.classList.add('error');
      return;
    }

    progressBarContainer.style.display = 'block';
    progressBar.style.width = '10%';
    progressLabel.textContent = 'Preparing upload...';
    statusDiv.textContent = 'Mempersiapkan upload file...';
    statusDiv.classList.add('info');
    
    // PERBAIKAN: Progress tracking yang lebih detail
    setTimeout(() => {
      progressBar.style.width = '30%';
      progressLabel.textContent = 'Forecast file...';
      statusDiv.textContent = 'Memproses forecast file...';
    }, 500);
  
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
      setTimeout(() => { 
        progressBar.style.width = '60%'; 
        progressLabel.textContent = 'Processing...'; 
        statusDiv.textContent = 'Processing forecast...'; 
      }, 400);
      
      // PERBAIKAN: Tambahkan timeout yang lebih panjang untuk dataset besar
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 7200000); // 2 jam timeout (7200 detik)
      
      console.log('🚀 Starting forecast request with 2-hour timeout...');
      
      // PERBAIKAN: Retry mechanism untuk handle network issues
      let response;
      let retryCount = 0;
      const maxRetries = 3;
      
      while (retryCount < maxRetries) {
        try {
          console.log(`🔄 Attempt ${retryCount + 1}/${maxRetries}...`);
          
          response = await fetch('/process-forecast', {
            method: 'POST',
            body: formData,
            signal: controller.signal
          });
          
                clearTimeout(timeoutId); // Clear timeout jika berhasil
      console.log('✅ Request successful on attempt', retryCount + 1);
      
      // PERBAIKAN: Jika response timeout tapi backend masih proses, mulai long polling
      if (response.status === 202 || response.status === 200) {
        console.log('📡 Response received, starting status monitoring...');
        break;
      }
      
      break;
          
        } catch (fetchError) {
          retryCount++;
          console.warn(`⚠️ Attempt ${retryCount} failed:`, fetchError.message);
          
          if (retryCount >= maxRetries) {
            clearTimeout(timeoutId);
            throw fetchError; // Re-throw error jika semua retry gagal
          }
          
          // Wait before retry
          console.log(`⏳ Waiting 2 seconds before retry...`);
          await new Promise(resolve => setTimeout(resolve, 2000));
          
          // Update progress
          progressBar.style.width = `${30 + (retryCount * 10)}%`;
          progressLabel.textContent = `Retry ${retryCount}/${maxRetries}...`;
          statusDiv.textContent = `Forecast gagal, mencoba lagi... (${retryCount}/${maxRetries})`;
        }
      }
      
      progressBar.style.width = '90%';
      progressLabel.textContent = 'Finalizing...';
      statusDiv.textContent = 'Finalizing...';
      
      if (!response.ok) {
        const errText = await response.text();
        statusDiv.textContent = 'Gagal memproses: ' + errText;
        statusDiv.className = '';
        statusDiv.classList.add('error');
        progressBarContainer.style.display = 'none';
        return;
      }
      
      const data = await response.json();
      console.log('Response data:', data);
      
      if (data.status !== 'success' || !data.file_id) {
        statusDiv.textContent = 'Gagal proses: ' + (data.message || 'Unknown error');
        statusDiv.className = '';
        statusDiv.classList.add('error');
        progressBarContainer.style.display = 'none';
        return;
      }

      // PERBAIKAN: Handle encrypted data
      if (data.encrypted_data && data.session_id) {
        console.log('Received encrypted data, decrypting...');
        
        // Dekripsi data menggunakan endpoint khusus
        try {
          const decryptResponse = await fetch('/api/dashboard-data', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              session_id: data.session_id,
              encrypted_data: data.encrypted_data
            })
          });
          
          if (!decryptResponse.ok) {
            throw new Error('Failed to decrypt dashboard data');
          }
          
          const decryptData = await decryptResponse.json();
          
          if (decryptData.status === 'success' && decryptData.data) {
            originalData = decryptData.data.original_df || [];
            backtestData = decryptData.data.forecast_df || [];
            realtimeData = decryptData.data.real_time_forecast || [];
            totalStats = decryptData.data.total_stats || {}; // Load total stats
            
            console.log('Data decrypted successfully:', {
              originalData: originalData.length,
              backtestData: backtestData.length,
              realtimeData: realtimeData.length,
              totalStats: totalStats
            });
            
            // Render dashboard dengan data baru
            console.log('🔄 Rendering dashboards with new data...');
            setTimeout(() => {
              renderRealtimeDashboard();
              renderBacktestDashboard();
            }, 100);
  } else {
            throw new Error('Invalid decrypted data format');
          }
          
        } catch (decryptError) {
          console.error('Error decrypting data:', decryptError);
          statusDiv.textContent = 'Gagal memproses data dashboard: ' + decryptError.message;
          statusDiv.className = '';
          statusDiv.classList.add('error');
          progressBarContainer.style.display = 'none';
          return;
        }
      } else {
        // Fallback untuk data tidak terenkripsi (legacy)
        console.log('No encrypted data found, using legacy format');
        if (data.original_df && data.forecast_df && data.real_time_forecast) {
          originalData = data.original_df;
          backtestData = data.forecast_df;
          realtimeData = data.real_time_forecast;
        } else {
          console.error('Missing required data in response:', data);
          statusDiv.textContent = 'Gagal proses: Data tidak lengkap dari server';
          statusDiv.className = '';
          statusDiv.classList.add('error');
          progressBarContainer.style.display = 'none';
          return;
        }
      }
      
      // Reset filter part number dan render ulang untuk menampilkan semua data
      if (partnoInput) {
        partnoInput.value = '';
        partnoInput.disabled = false;
        partnoInput.readOnly = false;
        partnoInput.style.pointerEvents = 'auto';
        partnoInput.style.cursor = 'text';
      }
      
      console.log('Forecast completed, updating dashboard with new data:', {
        originalData: originalData.length,
        realtimeData: realtimeData.length,
        backtestData: backtestData.length
      });
      
      // DEBUG: Log realtime data structure
      if (realtimeData.length > 0) {
        console.log('🔍 DEBUG: realtimeData columns:', Object.keys(realtimeData[0]));
        console.log('🔍 DEBUG: Sample realtime data:', realtimeData.slice(0, 3));
        console.log('🔍 DEBUG: FORECAST_NEUTRAL values:', realtimeData.map(d => d.FORECAST_NEUTRAL).slice(0, 10));
      }
      
      // Render ulang dashboard dengan delay untuk memastikan Chart.js tersedia
      setTimeout(() => {
        console.log('Rendering dashboard with new forecast data...');
        renderRealtimeDashboard();
        renderBacktestDashboard();
      }, 100);
      
      // Reset month filters
      selectedRealtimeMonths = [];
      selectedBacktestMonths = [];
      
      // Pastikan month picker input tidak disabled
      const realtimeMonthInput = document.getElementById('realtime-month-picker');
      const backtestMonthInput = document.getElementById('backtest-month-picker');
      
      if (realtimeMonthInput) {
        realtimeMonthInput.disabled = false;
        realtimeMonthInput.readOnly = true; // Tetap readonly untuk flatpickr
        realtimeMonthInput.style.pointerEvents = 'auto';
        realtimeMonthInput.style.cursor = 'pointer';
      }
      
      if (backtestMonthInput) {
        backtestMonthInput.disabled = false;
        backtestMonthInput.readOnly = true; // Tetap readonly untuk flatpickr
        backtestMonthInput.style.pointerEvents = 'auto';
        backtestMonthInput.style.cursor = 'pointer';
      }
      
      setupMonthPickers();
      // Render dashboard dengan data real (semua part number)
      renderRealtimeDashboard();
      renderBacktestDashboard();

      const fileId = data.file_id;
      progressBar.style.width = '100%';
      progressLabel.textContent = 'Forecast selesai, file diunduh';
      statusDiv.textContent = 'Forecast selesai, file diunduh.';
      statusDiv.className = '';
      statusDiv.classList.add('success');
      
      const downloadUrl = `/download-forecast?file_id=${encodeURIComponent(fileId)}`;
      console.log('Downloading file from:', downloadUrl);
      const blobResp = await fetch(downloadUrl);
      
      if (!blobResp.ok) {
        const errorText = await blobResp.text();
        console.error('Download failed:', blobResp.status, errorText);
        statusDiv.textContent = 'Gagal download file hasil: ' + errorText;
        statusDiv.className = '';
        statusDiv.classList.add('error');
        return;
      }
  
      const blob = await blobResp.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'forecast_result.xlsx';
      document.body.appendChild(a);
      a.click();
      
      setTimeout(() => {
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        progressBarContainer.style.display = 'none';
        progressBar.style.width = '0%';
        progressLabel.textContent = '';
      }, 1200);
      
      // Log success
      console.log('File downloaded successfully:', fileId);
  
    } catch (err) {
      console.error('❌ Error during forecast processing:', err);
      
      // PERBAIKAN: Handle berbagai jenis error dengan lebih detail
      let errorMessage = err.message;
      
      if (err.name === 'AbortError') {
        errorMessage = 'Request timeout setelah 2 jam. Dataset terlalu besar atau server sedang sibuk. Silakan coba lagi.';
        console.log('⏰ Request timeout - dataset mungkin terlalu besar');
      } else if (err.message.includes('Failed to fetch')) {
        errorMessage = 'Koneksi ke server gagal. Periksa koneksi internet Anda atau coba refresh halaman.';
        console.log('🌐 Network connection failed');
      } else if (err.message.includes('NetworkError')) {
        errorMessage = 'Error jaringan. Server mungkin sedang sibuk atau koneksi terputus.';
        console.log('🌐 Network error detected');
      } else {
        errorMessage = `Gagal proses: ${err.message}`;
        console.log('❌ Other error:', err.name, err.message);
      }
      
      // Update UI dengan error yang lebih informatif
      progressBarContainer.style.display = 'none';
      statusDiv.textContent = errorMessage;
      statusDiv.className = '';
      statusDiv.classList.add('error');
      
      // Log error details untuk debugging
      console.log('Error details:', {
        name: err.name,
        message: err.message,
        stack: err.stack
      });
      
      // Tetap render dashboard dengan data yang ada
      renderRealtimeDashboard();
      renderBacktestDashboard();
    }
  });
}