async function getFollowersSorted(username) {

const followers = [];

let offset = 0, page;

do {

const res = await fetch(`https://api.scratch.mit.edu/users/${encodeURIComponent(username)}/followers?limit=40&offset=${offset}`);

page = await res.json();

followers.push(...page.map(f => f.username));

offset += 40;

} while (page.length > 0);

console.log(`Found ${followers.length} followers`);

async function getFollowerCount(user) {

let count = 0;

let offset = 0;

let page;

do {

const res = await fetch(`https://api.scratch.mit.edu/users/${encodeURIComponent(user)}/followers?limit=40&offset=${offset}`);

if (!res.ok) return 0;

page = await res.json();

count += page.length;

offset += 40;

} while (page.length > 0);

return count;

}

// You can make the batch size bigger (20 would probably work fine, tbh) for it to run faster, or lower the delay as well, 500 or so would probs be chill. You should also keep in mind, however, the faster you make it, the more likely Scratch is to get angry at you, and also the less likely a request is to properly get through, so it might actually slow stuff down

const batchSize = 5;

const delayBetweenBatches = 1000;

const results = [];

for (let i = 0; i < followers.length; i += batchSize) {

const chunk = followers.slice(i, i + batchSize);

const data = await Promise.all(chunk.map(async f => {

const count = await getFollowerCount(f);

return { username: f, followers: count };

}));

results.push(...data);

if (i + batchSize < followers.length) {

await new Promise(r => setTimeout(r, delayBetweenBatches));

}

}

results.sort((a, b) => b.followers - a.followers);

console.table(results);

const csv = "username,followers\n" + results.map(r => `${r.username},${r.followers}`).join("\n");

const blob = new Blob([csv], { type: "text/csv" });

const url = URL.createObjectURL(blob);

const a = document.createElement("a");

a.href = url;
a.download `${username}_followers_sorted.csv`; // change this to download, Reddit wouldn't let me post it otherwise

document.body.appendChild(a);

a.click();

a.remove();

URL.revokeObjectURL(url);

return results;

}

// Just change this bit to the username of the person you want, hit enter and you're all set!

getFollowersSorted("Tri-Bunny");