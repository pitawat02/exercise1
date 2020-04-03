text = 'this is from js';
let names = abc.join(", ");
let p = document.getElementById("text1");
let p2 = document.getElementById("text2");
let p3 = document.getElementById("text3");
p4 = document.querySelector(".main p.p4")
pall = document.querySelectorAll(".main p")

p.textContent = text;
p2.textContent = names;
p3.textContent = def;
p4.textContent = def

for(var i = 0; i<pall.length ; i++){
    pall[i].textContent += " addinloop";
}

//--------------------------------//
var main2 = document.querySelector(".main2");
var h2 = main2.firstElementChild;
var ul = h2.nextElementSibling;
var ul2 = main2.children[2];
var li = ul.children;

h2.textContent += " askjdlasdj"
li[0].textContent += " a132"
li[2].firstChild.nodeValue = li[2].firstChild.nodeValue.toUpperCase();

let pointm = document.querySelector('.main2')
let pointli = pointm.querySelectorAll("ul li")
for(var i = 0; i<pointli.length ; i++){
    pointli[i].textContent += pointli[i].dataset.year;
}
//ul2.textContent += " askjdlasdj"
bool = main2.contains(ul)

//-------------------------------------//
let main3 = document.querySelector('.main3');
let annn = main3.querySelector('h2') ;
annn.textContent = 'Loading Movies' ;
let pee = main3.querySelector('p') ;
pee.innerHTML = 'And now a list of <strong>MOVIES</strong>';
let unlist = document.createElement('ul') ;
main3.appendChild (unlist) ;

abc.forEach(function(item) {
let li = document.createElement('li');
let txt = document.createTextNode(item);
li.appendChild(txt);
unlist.appendChild(li);
})

//main.removeChild(ul);