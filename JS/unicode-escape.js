// Unicode Characters in JavaScript
// Escape Sequences in JavaScript
// http://www.unicode.org/charts/

// String.fromCharCode(num [, num, num])
// myString.charCodeAt(index)
//  \u0434
// \0 \' \" \\ \n \r \t 
//

let yesRU =     '\u0434\u0430'; // Russian - yes
let milkDK =    'm\u00E6lk';       // milk
let breadNO =   'br\u00F8d';       // bread
let tomorrowES = 'ma\u00F1ana';     // tomorrow
let emojiJP =   '\u3047\u3082\u3058';  // Hiragana - emoji

let log = console.log;

log('\'\\a\t\ta\n\ta')

log(yesRU);
log(milkDK);
log(breadNO);
log(tomorrowES);
log(emojiJP);
  
log(milkDK.charCodeAt(0));

let s = String.fromCharCode(0x0434);
log(s);

//  العربية
