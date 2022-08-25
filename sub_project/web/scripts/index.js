let NamecardType = "";
//These are a functions of the button used to select a theme.

//Move to Namecard_modern.html.
let ModernButton = document.querySelector("#modern__select");
ModernButton.addEventListener("click", () => {
  NamecardType = "modern";
  console.log(NamecardType);
  window.location.href = "../page/Namecard_modern.html";
})

//Move to Namecard_simple.html.
let SimpleButton = document.querySelector("#simple__select");
SimpleButton.addEventListener("click", () => {
  NamecardType = "simple";
  console.log(NamecardType);
  window.location.href = "../page/Namecard_simple.html";
})

//Move to Namecard_colorful.html.
let ColorButton = document.querySelector("#color__select");
ColorButton.addEventListener("click", () => {
  NamecardType = "colorful";
  console.log(NamecardType);
  window.location.href = "../page/Namecard_colorful.html";
})

//If you want to add another theme, change "XXXX" of the code below.

/*let XXXXButton = document.querySelector("#color__select");
XXXXButton.addEventListener("click", () => {
  NamecardType = "XXXX";
  console.log(NamecardType);
  window.location.href = "../page/Namecard_XXXX.html";
})*/