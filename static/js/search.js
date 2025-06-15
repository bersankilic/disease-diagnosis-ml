// Arama ve filtreleme JavaScript fonksiyonları
const SymptomSearch = {
  allSymptoms: [],

  init: function (symptoms) {
    this.allSymptoms = symptoms;
    this.renderSymptoms(symptoms);
    this.updateFilteredCount(symptoms.length);
  },

  renderSymptoms: function (symptoms) {
    const grid = document.getElementById("symptomGrid");
    grid.innerHTML = "";

    symptoms.forEach((symptom, index) => {
      const div = document.createElement("div");
      div.className = "symptom-item";
      div.innerHTML = `
                <input
                    type="checkbox"
                    id="symptom_${index + 1}"
                    name="symptoms"
                    value="${symptom.english}"
                    onchange="updateSelectedCount()"
                />
                <label for="symptom_${index + 1}">
                    ${symptom.turkish}
                </label>
            `;
      grid.appendChild(div);
    });
  },

  filter: function () {
    const searchText = document
      .getElementById("symptomSearch")
      .value.toLowerCase();

    if (searchText === "") {
      this.renderSymptoms(this.allSymptoms);
      this.updateFilteredCount(this.allSymptoms.length);
      return;
    }

    const filteredSymptoms = this.allSymptoms.filter(
      (symptom) =>
        symptom.turkish.toLowerCase().includes(searchText) ||
        symptom.english.toLowerCase().includes(searchText)
    );

    this.renderSymptoms(filteredSymptoms);
    this.updateFilteredCount(filteredSymptoms.length);
  },

  clearSearch: function () {
    document.getElementById("symptomSearch").value = "";
    this.renderSymptoms(this.allSymptoms);
    this.updateFilteredCount(this.allSymptoms.length);
  },

  updateFilteredCount: function (count) {
    document.getElementById("filteredCount").textContent = count;
  },
};

// Global fonksiyonlar
function filterSymptoms() {
  SymptomSearch.filter();
}

function clearSearch() {
  SymptomSearch.clearSearch();
}

function updateSelectedCount() {
  const checkedBoxes = document.querySelectorAll(
    'input[name="symptoms"]:checked'
  );
  document.getElementById("selectedCount").textContent = checkedBoxes.length;
}

function clearSymptoms() {
  const checkboxes = document.querySelectorAll('input[name="symptoms"]');
  checkboxes.forEach((cb) => (cb.checked = false));
  updateSelectedCount();
}
