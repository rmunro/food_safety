

// GLOBAL VARIABLES
var world_map;                        // this is the leaflet map object for the world
var europe_map;                        // the leaflet map object for europe
var world_countries_layer;            // leaflet layer object for the world countries
var europe_countries_layer;            // leaflet layer object for europe countries (data actually contain all counties, map is just limited to the europe region
var selected_europe_country;        // the europe country that is currently highlighted on the map and selected in the radio button group
var selected_world_countries = [];    // list of all currently selected world countries

 
async function save_report(){       
    hazard = document.getElementById('hazard').value;
    food = document.getElementById('food').value;
    origin = document.getElementById('origin').value;
    destination = document.getElementById('destination').value;
    report_id = document.getElementById('report_id').value;
    report_date = document.getElementById('report_date').value;
    report = document.getElementById('report_text').value;
    
    date = "";
    
    
    saved = await eel.save_report([report_id, report, report_date, food, hazard, origin, destination])(); // Call to python function
    
    get_next_report();
}

async function get_next_report(){   
   
    // GET NEXT REPORT
    report = await eel.get_next_report()(); // Call to python function

    report_id = report[0];    
    report_text = report[1];
    report_date = report[2];

    
    document.getElementById('report_id').value = report_id;    
    document.getElementById('report_text').value = report_text;    
    document.getElementById('annotation').innerHTML = report_text;
    document.getElementById('report_date').value = report_date;    

    document.getElementById('hazard').value = "";
    document.getElementById('food').value = "";
    document.getElementById('origin').value = "";
    document.getElementById('destination').value = "";

    document.getElementById('hazard').focus(); 

    get_candidate_spans(report_text); // add autocomplete for hazards
    
}

async function get_candidate_spans(text){    
    candidates = await eel.get_candidate_spans(text)(); // Call to python function
    
    console.log(candidates);
    
    hazard = candidates['Hazard'];
    food = candidates['Food'];
    origin = candidates['Origin'];
    destination = candidates['Destination'];
    
    // add autocomplete for potential hazards
    
    console.log(hazard);

    autocomplete(document.getElementById("hazard"), hazard);
    autocomplete(document.getElementById("food"), food);
    autocomplete(document.getElementById("origin"), origin);
    autocomplete(document.getElementById("destination"), destination);
}


var current_details = "";



async function get_recent_reports(max=10){
    origin = document.getElementById("origin").value;
    hazard = document.getElementById("hazard").value;
    food = document.getElementById("food").value;
    
    combined = origin+hazard+food;
    if(combined != current_details){
        current_details = combined;
        reports = await eel.get_recent_reports(origin, hazard, food)();
        
        console.log(combined);
        
        if(reports.length > 0){
            reports_div = document.getElementById("relevant_reports");
            reports_div.innerHTML = "<h4>Relevant Recent Reports:</h4>";                
            for(var i=0; i < reports.length; i++){
                report_date = reports[i][0]
                report_text = reports[i][1];
                
                if(origin.length > 0){
                    report_text = report_text.replace(origin,"<b>"+origin+"</b>");
                }
                if(hazard.length > 0){
                    report_text = report_text.replace(hazard,"<b>"+hazard+"</b>");
                }
                if(food.length > 0){
                    report_text = report_text.replace(food,"<b>"+food+"</b>");
                }
                reports_div.innerHTML += "<p>"+report_date+": "+report_text+"</p>";
            }
        }
    }
    
    setTimeout(get_recent_reports, 2000);

     
}

setTimeout(get_recent_reports, 5000);





/*
AUTO-COMPLETE CODE SNIPPET BELOW FROM:
https://www.w3schools.com/howto/howto_js_autocomplete.asp

*/

function autocomplete(inp, arr) {
  /*the autocomplete function takes two arguments,
  the text field element and an array of possible autocompleted values:*/
  var currentFocus;
  /*execute a function when someone writes in the text field:*/
  inp.addEventListener("input", function(e) {
      var a, b, i, val = this.value;
      /*close any already open lists of autocompleted values*/
      closeAllLists();
      if (!val) { return false;}
      currentFocus = -1;
      /*create a DIV element that will contain the items (values):*/
      a = document.createElement("DIV");
      a.setAttribute("id", this.id + "autocomplete-list");
      a.setAttribute("class", "autocomplete-items");
      /*append the DIV element as a child of the autocomplete container:*/
      this.parentNode.appendChild(a);
      /*for each item in the array...*/
      for (i = 0; i < arr.length; i++) {
        /*check if the item starts with the same letters as the text field value:*/
        if (arr[i].substr(0, val.length).toUpperCase() == val.toUpperCase()) {
          /*create a DIV element for each matching element:*/
          b = document.createElement("DIV");
          /*make the matching letters bold:*/
          b.innerHTML = "<strong>" + arr[i].substr(0, val.length) + "</strong>";
          b.innerHTML += arr[i].substr(val.length);
          /*insert a input field that will hold the current array item's value:*/
          b.innerHTML += "<input type='hidden' value='" + arr[i] + "'>";
          /*execute a function when someone clicks on the item value (DIV element):*/
              b.addEventListener("click", function(e) {
              /*insert the value for the autocomplete text field:*/
              inp.value = this.getElementsByTagName("input")[0].value;
              /*close the list of autocompleted values,
              (or any other open lists of autocompleted values:*/
              closeAllLists();
          });
          a.appendChild(b);
        }
      }
  });
  /*execute a function presses a key on the keyboard:*/
  inp.addEventListener("keydown", function(e) {
      var x = document.getElementById(this.id + "autocomplete-list");
      if (x) x = x.getElementsByTagName("div");
      if (e.keyCode == 40) {
        /*If the arrow DOWN key is pressed,
        increase the currentFocus variable:*/
        currentFocus++;
        /*and and make the current item more visible:*/
        addActive(x);
      } else if (e.keyCode == 38) { //up
        /*If the arrow UP key is pressed,
        decrease the currentFocus variable:*/
        currentFocus--;
        /*and and make the current item more visible:*/
        addActive(x);
      } else if (e.keyCode == 13) {
        /*If the ENTER key is pressed, prevent the form from being submitted,*/
        e.preventDefault();
        if (currentFocus > -1) {
          /*and simulate a click on the "active" item:*/
          if (x) x[currentFocus].click();
        }
      }
  });
  function addActive(x) {
    /*a function to classify an item as "active":*/
    if (!x) return false;
    /*start by removing the "active" class on all items:*/
    removeActive(x);
    if (currentFocus >= x.length) currentFocus = 0;
    if (currentFocus < 0) currentFocus = (x.length - 1);
    /*add class "autocomplete-active":*/
    x[currentFocus].classList.add("autocomplete-active");
  }
  function removeActive(x) {
    /*a function to remove the "active" class from all autocomplete items:*/
    for (var i = 0; i < x.length; i++) {
      x[i].classList.remove("autocomplete-active");
    }
  }
  function closeAllLists(elmnt) {
    /*close all autocomplete lists in the document,
    except the one passed as an argument:*/
    var x = document.getElementsByClassName("autocomplete-items");
    for (var i = 0; i < x.length; i++) {
      if (elmnt != x[i] && elmnt != inp) {
      x[i].parentNode.removeChild(x[i]);
    }
  }
}
/*execute a function when someone clicks in the document:*/
document.addEventListener("click", function (e) {
    closeAllLists(e.target);
});
}



        