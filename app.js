/*
  Webcam Rock Paper Scissors — app.js
  1. Collect webcam photos of your hand for each move
  2. Train a small CNN in the browser with TensorFlow.js
  3. Play against an AI that predicts your next move with a Markov model
*/
const NAMES=['Rock','Paper','Scissors'], EMOJI=['✊','✋','✌️'], COLORS=['--rock','--paper','--scissors'];
const S=64, MAX_PER_CLASS=200;
const $=id=>document.getElementById(id);
const video=$('video');
const grab=document.createElement('canvas'); grab.width=grab.height=S;
const gctx=grab.getContext('2d',{willReadFrequently:true});
const samples=[[],[],[]];
let model=null, predictTimer=null, recent=[], busy=false;

/* ---------- camera ---------- */
$('startCam').onclick=async()=>{
  $('camErr').hidden=true;
  try{
    const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:640},height:{ideal:640}},audio:false});
    video.srcObject=stream; await video.play();
    $('camOverlay').style.display='none'; $('frame').hidden=false;
    $('p1').classList.remove('off');
  }catch(e){
    $('camErr').textContent=(e&&e.name==='NotAllowedError')
      ? 'Camera access was blocked. Allow the camera for this page in your browser settings, then press Start camera again. If it still fails, open this page in its own tab.'
      : 'No camera could be opened ('+(e&&e.name||'unknown error')+'). Check that a webcam is connected and not in use by another app.';
    $('camErr').hidden=false;
  }
};

/* Crop the middle 80% square of the video (matches the dashed frame) into an SxS RGB array */
function snapshot(){
  const vw=video.videoWidth, vh=video.videoHeight; if(!vw) return null;
  const side=Math.min(vw,vh)*0.8, sx=(vw-side)/2, sy=(vh-side)/2;
  gctx.drawImage(video,sx,sy,side,side,0,0,S,S);
  const d=gctx.getImageData(0,0,S,S).data, out=new Float32Array(S*S*3);
  for(let i=0,j=0;i<d.length;i+=4){out[j++]=d[i]/255;out[j++]=d[i+1]/255;out[j++]=d[i+2]/255;}
  return out;
}

/* ---------- step 1: collect ---------- */
let recTimer=null, recClass=-1;
function startRec(c){
  if(recTimer||!video.videoWidth) return;
  recClass=c; $('rec').style.display='block';
  document.querySelector(`.move[data-c="${c}"]`).classList.add('active');
  recTimer=setInterval(()=>{
    if(samples[c].length>=MAX_PER_CLASS) return;
    const s=snapshot(); if(s){samples[c].push(s); updateCounts();}
  },90);
}
function stopRec(){
  if(!recTimer) return; clearInterval(recTimer); recTimer=null;
  $('rec').style.display='none';
  document.querySelectorAll('.move').forEach(b=>b.classList.remove('active'));
}
document.querySelectorAll('.move').forEach(b=>{
  const c=+b.dataset.c;
  b.addEventListener('pointerdown',e=>{e.preventDefault();startRec(c);});
  ['pointerup','pointerleave','pointercancel'].forEach(ev=>b.addEventListener(ev,stopRec));
  b.addEventListener('contextmenu',e=>e.preventDefault());
});
const KEYMAP={r:0,p:1,s:2};
addEventListener('keydown',e=>{if(e.repeat||$('p1').classList.contains('off'))return;const c=KEYMAP[e.key.toLowerCase()];if(c!==undefined)startRec(c);});
addEventListener('keyup',e=>{if(KEYMAP[e.key.toLowerCase()]!==undefined)stopRec();});

function updateCounts(){
  samples.forEach((s,i)=>$('c'+i).textContent=s.length+(s.length>=MAX_PER_CLASS?' (full)':' samples'));
  const ready=samples.every(s=>s.length>=20);
  $('trainBtn').disabled=!ready||busy;
  if(!ready) $('trainMsg').textContent='Collect at least 20 samples of each shape to unlock training.';
  else if(!model) $('trainMsg').textContent='A small convolutional network learns from your samples. Takes about 10–30 seconds.';
}
updateCounts();

/* ---------- step 2: train ---------- */
function buildModel(){
  const m=tf.sequential();
  m.add(tf.layers.conv2d({inputShape:[S,S,3],filters:16,kernelSize:3,padding:'same',activation:'relu'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.conv2d({filters:64,kernelSize:3,padding:'same',activation:'relu'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dropout({rate:0.4}));
  m.add(tf.layers.dense({units:64,activation:'relu'}));
  m.add(tf.layers.dense({units:3,activation:'softmax'}));
  m.compile({optimizer:tf.train.adam(0.001),loss:'categoricalCrossentropy',metrics:['accuracy']});
  return m;
}

/* Augment: each sample also appears mirrored with a random brightness shift */
function augmented(src){
  const out=new Float32Array(src.length), k=0.75+Math.random()*0.5;
  for(let y=0;y<S;y++)for(let x=0;x<S;x++){
    const a=(y*S+x)*3, b=(y*S+(S-1-x))*3;
    for(let ch=0;ch<3;ch++) out[b+ch]=Math.min(1,src[a+ch]*k);
  }
  return out;
}

$('trainBtn').onclick=async()=>{
  busy=true; $('trainBtn').disabled=true; stopPredicting();
  $('trainMsg').textContent='Training…'; $('acc').textContent=''; $('prog').style.width='0';
  // Hold out the LAST 15% of each shape's samples for testing (never augmented, never trained on),
  // so the accuracy score reflects photos the model hasn't seen.
  const train=[], val=[];
  samples.forEach((arr,c)=>{
    const cut=Math.floor(arr.length*0.85);
    arr.forEach((s,i)=>{ if(i<cut){train.push([s,c]);train.push([augmented(s),c]);} else val.push([s,c]); });
  });
  for(let i=train.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[train[i],train[j]]=[train[j],train[i]];}
  const toTensors=list=>{
    const flat=new Float32Array(list.length*S*S*3);
    list.forEach(([s],i)=>flat.set(s,i*S*S*3));
    return [tf.tensor4d(flat,[list.length,S,S,3]), tf.oneHot(tf.tensor1d(list.map(t=>t[1]),'int32'),3)];
  };
  const [xs,ys]=toTensors(train), [vx,vy]=toTensors(val);
  if(model) model.dispose();
  model=buildModel();
  const EPOCHS=15;
  try{
    await model.fit(xs,ys,{epochs:EPOCHS,batchSize:32,validationData:[vx,vy],shuffle:true,callbacks:{
      onEpochEnd:async(ep,logs)=>{
        $('prog').style.width=((ep+1)/EPOCHS*100)+'%';
        $('acc').textContent=`Epoch ${ep+1}/${EPOCHS}: ${(logs.val_acc*100).toFixed(0)}% accuracy on photos it never trained on`;
        await tf.nextFrame();
      }}});
    $('trainMsg').textContent='Trained. If the live read below looks shaky, add more varied samples and train again.';
    $('p3').classList.remove('off'); $('saveBtn').disabled=false; startPredicting();
  }catch(e){
    $('trainMsg').textContent='Training failed: '+e.message;
  }finally{
    xs.dispose(); ys.dispose(); vx.dispose(); vy.dispose(); busy=false; updateCounts();
  }
};

/* ---------- live prediction ---------- */
$('bars').innerHTML=NAMES.map((n,i)=>`<div class="bar"><span>${EMOJI[i]} ${n}</span><div class="t"><div id="b${i}" style="background:var(${COLORS[i]})"></div></div><span class="v" id="v${i}">0%</span></div>`).join('');
function startPredicting(){
  stopPredicting();
  predictTimer=setInterval(async()=>{
    const s=snapshot(); if(!s||!model) return;
    const p=tf.tidy(()=>model.predict(tf.tensor4d(s,[1,S,S,3])));
    const probs=Array.from(await p.data()); p.dispose();
    recent.push({t:performance.now(),probs,s}); if(recent.length>30) recent.shift();
    probs.forEach((v,i)=>{$('b'+i).style.width=(v*100)+'%';$('v'+i).textContent=Math.round(v*100)+'%';});
  },100);
}
function stopPredicting(){clearInterval(predictTimer);predictTimer=null;}

/* ---------- AI opponent: learns your patterns with a Markov model ---------- */
const hist=[], table={1:{},2:{},3:{}};
function aiPredict(){
  for(const order of [3,2,1]){
    if(hist.length<order) continue;
    const row=table[order][hist.slice(-order).join('')];
    if(!row) continue;
    const total=row[0]+row[1]+row[2];
    if(total<2) continue;
    const best=row.indexOf(Math.max(...row));
    return {move:best,why:`after your last ${order===1?'throw':order+' throws'}, you went ${NAMES[best].toLowerCase()} ${row[best]} of ${total} times`};
  }
  if(hist.length>=3){
    const f=[0,0,0]; hist.forEach(m=>f[m]++);
    const best=f.indexOf(Math.max(...f));
    return {move:best,why:`${NAMES[best].toLowerCase()} is your most common throw so far`};
  }
  return null;
}
function learn(move){
  hist.push(move);
  for(const order of [1,2,3]){
    if(hist.length<=order) continue;
    const key=hist.slice(-order-1,-1).join('');
    (table[order][key]??=[0,0,0])[move]++;
  }
}


/* After "Shoot!", ignore the first moment while your hand is still moving, then lock in
   the first move the model reads confidently on several frames in a row. */
async function readSteadyHand(){
  const GRACE=300, NEED=3, MIN_CONF=0.7, TIMEOUT=1800;
  const start=performance.now();
  let lastT=start+GRACE, streakMove=-1, streak=0;
  const seen=[];
  while(performance.now()-start<TIMEOUT){
    await sleep(40);
    for(const r of recent.filter(r=>r.t>lastT)){
      lastT=r.t; seen.push(r);
      const c=Math.max(...r.probs), m=r.probs.indexOf(c);
      if(c>=MIN_CONF && m===streakMove) streak++;
      else { streakMove=c>=MIN_CONF?m:-1; streak=streakMove<0?0:1; }
      if(streak>=NEED) return {you:m,conf:c,frame:r.s};
    }
  }
  // Never steady: fall back to the average of the last few frames
  const tail=seen.slice(-4);
  if(!tail.length) return {you:0,conf:0,frame:null};
  const avg=[0,0,0]; tail.forEach(r=>r.probs.forEach((v,i)=>avg[i]+=v/tail.length));
  const conf=Math.max(...avg);
  return {you:avg.indexOf(conf),conf,frame:tail[tail.length-1].s};
}

/* Draw the exact 64x64 photo the model judged, so you can check its call */
function showSeen(frame,label){
  if(!frame){ $('seen').hidden=true; return; }
  const ctx=$('seenCanvas').getContext('2d'), img=ctx.createImageData(S,S);
  for(let i=0,j=0;j<frame.length;i+=4){
    img.data[i]=frame[j++]*255; img.data[i+1]=frame[j++]*255; img.data[i+2]=frame[j++]*255; img.data[i+3]=255;
  }
  ctx.putImageData(img,0,0);
  $('seenText').textContent=label; $('seen').hidden=false;
}

/* ---------- game loop ---------- */
const score={w:0,d:0,l:0};
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
$('playBtn').onclick=async()=>{
  if(busy) return; busy=true; $('playBtn').disabled=true; $('trainBtn').disabled=true;
  $('youHand').textContent=''; $('aiHand').textContent=''; $('verdict').textContent=''; $('why').textContent=''; $('seen').hidden=true;
  // AI commits before seeing your hand
  const guess=Math.random()<0.1?null:aiPredict();
  const aiMove=guess?(guess.move+1)%3:Math.floor(Math.random()*3);
  for(const w of ['Rock','Paper','Scissors']){$('count').textContent=w;await sleep(550);}
  $('count').textContent='Shoot!';
  const {you,conf,frame}=await readSteadyHand();
  $('count').textContent='';
  showSeen(frame, conf<0.5 ? 'What the camera saw at “Shoot!”' : `What the model saw at “Shoot!”: ${NAMES[you]}, ${Math.round(conf*100)}% sure`);
  if(conf<0.5){
    $('verdict').textContent='Couldn’t read your hand';
    $('why').textContent='Your hand never held still long enough. Throw on “Shoot!” and hold the shape inside the square for a moment.';
  }else{
    $('youHand').textContent=EMOJI[you]; $('aiHand').textContent=EMOJI[aiMove];
    const r=(you-aiMove+3)%3;
    const res=r===0?'d':r===1?'w':'l';
    score[res]++;
    $('verdict').textContent=res==='w'?'You win':res==='l'?'AI wins':'Draw';
    $('verdict').style.color=res==='w'?'var(--scissors)':res==='l'?'var(--lose)':'var(--ink)';
    $('why').textContent=guess
      ? `The AI expected ${NAMES[guess.move].toLowerCase()}: ${guess.why}.`
      : 'The AI threw at random this round.';
    const h=document.createElement('i'); h.className=res; h.textContent=EMOJI[you]; h.title=`You: ${NAMES[you]}, AI: ${NAMES[aiMove]}`;
    $('history').appendChild(h);
    learn(you);
    $('sW').textContent=score.w; $('sD').textContent=score.d; $('sL').textContent=score.l;
  }
  busy=false; $('playBtn').disabled=false; updateCounts();
};
$('resetBtn').onclick=()=>{
  score.w=score.d=score.l=0; hist.length=0; table[1]={};table[2]={};table[3]={};
  ['sW','sD','sL'].forEach(id=>$(id).textContent='0'); $('history').innerHTML='';
  $('youHand').textContent=''; $('aiHand').textContent=''; $('verdict').textContent=''; $('why').textContent='';
};

/* ---------- save / load the trained model (browser storage) ---------- */
const MODEL_KEY='localstorage://rps-webcam-model';
$('saveBtn').onclick=async()=>{
  if(!model) return;
  try{ await model.save(MODEL_KEY); $('saveMsg').textContent='Saved in this browser.'; }
  catch(e){ $('saveMsg').textContent='Could not save: '+e.message; }
};
$('loadBtn').onclick=async()=>{
  try{
    const m=await tf.loadLayersModel(MODEL_KEY);
    if(model) model.dispose();
    model=m; $('saveBtn').disabled=false;
    $('saveMsg').textContent='Loaded your saved model.';
    $('p3').classList.remove('off');
    if(video.videoWidth) startPredicting();
    else $('saveMsg').textContent='Loaded. Start the camera to play.';
  }catch(e){ $('saveMsg').textContent='No saved model found yet. Train one and press Save model.'; }
};
// If the model was loaded before the camera started, begin predicting once video is live
video.addEventListener('playing',()=>{ if(model) startPredicting(); });