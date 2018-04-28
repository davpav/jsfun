import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  
  title = 'hello world';
  linmod : tf.Sequential;
  prediction:any;
  ngOnInit(){
    this.trinyourdragon();

  }
async trinyourdragon() {
this.linmod=tf.sequential();
this.linmod.add(tf.layers.dense({units: 1,inputShape: [1]}));
this.linmod.compile({loss: 'meanSquaredError',optimizer: 'sgd'});

const xs=tf.tensor1d(this.getRandomMap(88));
const ys=tf.tensor1d(this.getRandomMap(88));
await this.linmod.fit(xs,ys);
console.log("donE");

}

getRandomMap(length,min=0,max=10){
  let array=[];
  for(let i=0; i< length;i++){
    let current=Math.floor(Math.random() * max) + min;  
    array.push(current);
  }
  return array;
}

predict(value){
 
  const output = this.linmod.predict(tf.tensor2d([value],[1,1])) as any;
 this.prediction =Array.from(output.dataSync())[0];  
}

}
