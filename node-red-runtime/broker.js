// Local MQTT broker for classroom demo, LAN only; no authentication.
const net=require('net');const aedes=require('aedes')();
net.createServer(aedes.handle).listen(1883,'0.0.0.0',()=>console.log('VoltGuard MQTT broker listening on TCP 1883 (trusted LAN only)'));
aedes.on('client',c=>console.log('MQTT client:',c.id));
